import copy
import uuid
import re
import yaml
import openpyxl
import pptx
import docxtpl

from box import Box
from jinja2 import Template, Undefined, DebugUndefined, exceptions, Environment

from yamgam.state import state

# Custom Loader to merge instead of replace
class DeepMergeLoader(yaml.SafeLoader):
    def _recursive_merge(self, dict1, dict2):
        """Recursively merges dict2 into dict1."""
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                self._recursive_merge(dict1[key], value)
            else:
                dict1[key] = value
        return dict1

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            
            if key in mapping and isinstance(mapping[key], dict) and isinstance(value, dict):
                # Recursively merge the dictionaries
                self._recursive_merge(mapping[key], value)
            else:
                mapping[key] = value
        return mapping

# Register the constructor for all mappings
DeepMergeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    DeepMergeLoader.construct_mapping
)

class NullUndefined(Undefined):
    UNDEFINED = f"UNDEFINED:{uuid.uuid4().hex}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_object = kwargs.get('obj')

    def __getattr__(self, name):
        new_name = f"{self._undefined_name}.{name}" if self._undefined_name else name
        return NullUndefined(name=new_name)

    def __str__(self):
        return f"{NullUndefined.UNDEFINED}:{self._undefined_name}"

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False

class _PPTXRendering:
    table_data_pattern = re.compile(r"\{\{\s*table\s*:\s*([^}\s]+)\s*\}\}")

    def __init__(self, input_path, data):
        self.input_path = input_path
        self.model = data["model"]

        self.env = Environment()

        self.current_slide = None
        self.current_shape = None

        self.message_raw = []

    def process(self, output_path) -> str:
        ppt = pptx.Presentation(self.input_path)
        for slide in ppt.slides:
            self.current_slide = slide
            self._render_slide(slide)
        ppt.save(output_path)
        return "\n".join(self.message_raw)

    def _render_slide(self, slide):
        for shape in slide.shapes:
            self.current_shape = shape
            self._render_shape(shape)

    def _render_shape(self, shape):
        if shape.has_text_frame:
            self._render_text_frame(shape.text_frame)
        if shape.has_table:
            self._render_table(shape.table)

    def _render_text_frame(self, text_frame):
        for paragraph in text_frame.paragraphs:
            self._render_paragraph(paragraph)

    def _merge_placeholder_runs(self, paragraph):
        runs = paragraph.runs
        i = 0
        while i < len(runs):
            text = runs[i].text or ""
            if "{{" in text:
                # Found '{{', start merging
                merged = text
                j = i + 1
                # Keep adding subsequent runs until '}}' found or end reached
                while j < len(runs) and "}}" not in merged:
                    merged += runs[j].text or ""
                    j += 1
                if "}}" in merged:
                    # Merge successful: put merged content in the i-th run
                    runs[i].text = merged
                    # Clear text in intermediate runs while preserving their formatting
                    for k in range(i + 1, j):
                        runs[k].text = ""
                    # Skip processed segments
                    i = j
                else:
                    # Exit if no closing tag found
                    break
            else:
                i += 1

    def _render_paragraph(self, paragraph):
        # 1. First merge {{...}} across runs
        self._merge_placeholder_runs(paragraph)
        # 2. Then render by run (original logic)
        for run in paragraph.runs:
            self._render_run(run)

    def _render_table(self, table):
        self._prepare_table(table)
        for cell in table.iter_cells():
            self._render_cell(cell)

    def _prepare_table(self, table):
        for cell in table.iter_cells():
            matches = re.findall(self.table_data_pattern, cell.text)
            if matches:
                table_data_key = matches[0].strip()
                self._remove_row(table, 0)
                if table_data := self.model.get(table_data_key):
                    for i, row in enumerate(table_data):
                        self._add_row(table, i, table_data_key)
                    self._remove_row(table, 1)
            break

    def _render_cell(self, cell):
        self._render_text_frame(cell.text_frame)

    def _render_run(self, run):
        try:
            template = self.env.from_string(str(run.text))
            rendered = template.render(self.model)
        except exceptions.UndefinedError as error:
            error_text = f"{error.__class__.__name__}: {error}"
            self.message_raw.append(error_text)
        except exceptions.TemplateSyntaxError as error:
            error_text = (
                f"{error.__class__.__name__}: {error}\n"
                "you should re-write the whole {{}} tag"
            )
            self.message_raw.append(error_text)
        else:
            run.text = rendered

    @staticmethod
    def _add_row(table, row_index, table_data_key):
        new_row = copy.deepcopy(table._tbl.tr_lst[-1])

        for cell_index, tc in enumerate(new_row.tc_lst):
            cell = pptx.table._Cell(tc, new_row.tc_lst)
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.text = f"{{{{ {table_data_key}[{row_index}][{cell_index}] }}}}"

        table._tbl.append(new_row)

    @staticmethod
    def _remove_row(table, row_index):
        row = list(table.rows)[row_index]
        table._tbl.remove(row._tr)

def xlsx_file(input_path: str, output_path: str, context: Box):
    wb = openpyxl.load_workbook(input_path)

    for sheet in wb.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    template = Template(cell.value)
                    cell.value = template.render(context)

    if output_path:
        wb.save(output_path)

def docx_file(input_path: str, output_path: str, context: Box):
    doc = docxtpl.DocxTemplate(input_path)
    doc.render(context)
    if output_path:
        doc.save(output_path)

def pptx_file(input_path: str, output_path: str, context: Box):
    rendering = _PPTXRendering(input_path, context, output_path)
    if output_path:
        message = rendering.process(output_path)
        if state.verbose: print(message)

def raw_yaml(raw: str, context: object = {}) -> str:
    template = Template(raw, undefined=NullUndefined)
    print(Box(context))
    rendered = template.render(Box(context))
    return yaml.load(rendered, Loader=DeepMergeLoader)

def recursive_yaml(raw: str, depth: int) -> Box:
    previous = None
    current = {}
    count = 0
    while previous != current and count < depth:
        previous = current
        current = raw_yaml(raw, current)
        count += 1
    return Box(current)