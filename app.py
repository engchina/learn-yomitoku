import os
from pathlib import Path

import cv2
import gradio as gr

from yomitoku.constants import SUPPORT_OUTPUT_FORMAT
from yomitoku.data.functions import load_image, load_pdf
from yomitoku.document_analyzer import DocumentAnalyzer
from yomitoku.utils.logger import set_logger

logger = set_logger(__name__, "INFO")


def process_file(file, outdir, format, vis, device, td_cfg, tr_cfg, lp_cfg, tsr_cfg, ignore_line_break, figure,
                 figure_letter, figure_width, figure_dir):
    """
    Gradio 处理函数，用于处理上传的文件。
    """
    try:
        # 创建输出目录
        os.makedirs(outdir, exist_ok=True)
        logger.info(f"Output directory: {outdir}")

        # 解析配置
        configs = {
            "ocr": {
                "text_detector": {"path_cfg": td_cfg if td_cfg else None},
                "text_recognizer": {"path_cfg": tr_cfg if tr_cfg else None},
            },
            "layout_analyzer": {
                "layout_parser": {"path_cfg": lp_cfg if lp_cfg else None},
                "table_structure_recognizer": {"path_cfg": tsr_cfg if tsr_cfg else None},
            },
        }

        analyzer = DocumentAnalyzer(
            configs=configs,
            visualize=vis,
            device=device,
        )

        path = Path(file.name)
        format = format.lower()
        if format not in SUPPORT_OUTPUT_FORMAT:
            raise ValueError(f"Invalid output format: {format}")

        if format == "markdown":
            format = "md"

        # 加载文件
        if path.suffix[1:].lower() == "pdf":
            imgs = load_pdf(path)
        else:
            imgs = [load_image(path)]

        # 处理每页
        for page, img in enumerate(imgs):
            results, ocr, layout = analyzer(img)

            filename = path.stem
            if ocr is not None:
                ocr_path = os.path.join(outdir, f"{filename}_p{page + 1}_ocr.jpg")
                cv2.imwrite(ocr_path, ocr)

            if layout is not None:
                layout_path = os.path.join(outdir, f"{filename}_p{page + 1}_layout.jpg")
                cv2.imwrite(layout_path, layout)

            out_path = os.path.join(outdir, f"{filename}_p{page + 1}.{format}")
            if format == "json":
                results.to_json(out_path, ignore_line_break=ignore_line_break)
            elif format == "csv":
                results.to_csv(out_path, ignore_line_break=ignore_line_break)
            elif format == "html":
                results.to_html(out_path, ignore_line_break=ignore_line_break, img=img, export_figure=figure,
                                export_figure_letter=figure_letter, figure_width=figure_width, figure_dir=figure_dir)
            elif format == "md" or format == "markdown":
                results.to_markdown(out_path, ignore_line_break=ignore_line_break, img=img, export_figure=figure,
                                    export_figure_letter=figure_letter, figure_width=figure_width,
                                    figure_dir=figure_dir)

        return f"Processing completed. Files saved in {outdir}."
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return f"Error: {str(e)}"


# 创建 Gradio 接口
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Document Analyzer UI")

        with gr.Row():
            file_input = gr.File(label="Upload File", type="filepath")
            format_input = gr.Dropdown(
                choices=SUPPORT_OUTPUT_FORMAT,
                value="json",
                label="Output Format"
            )
            output_dir = gr.Textbox(value="results", label="Output Directory")
            visualize = gr.Checkbox(value=True, label="Visualize Results")
            device = gr.Radio(choices=["cuda", "cpu"], value="cuda", label="Device (cuda/cpu)")

        with gr.Row(visible=False):
            td_cfg = gr.Textbox(value="", label="Text Detector Config File")
            tr_cfg = gr.Textbox(value="", label="Text Recognizer Config File")
            lp_cfg = gr.Textbox(value="", label="Layout Parser Config File")
            tsr_cfg = gr.Textbox(value="", label="Table Structure Recognizer Config File")

        with gr.Row(visible=False):
            ignore_line_break = gr.Checkbox(value=False, label="Ignore Line Breaks")
            figure = gr.Checkbox(value=False, label="Export Figures")
            figure_letter = gr.Checkbox(value=False, label="Export Letters in Figures")
            figure_width = gr.Number(value=200, label="Figure Width")
            figure_dir = gr.Textbox(value="figures", label="Figure Directory")

        output = gr.Textbox(label="Processing Status")
        process_button = gr.Button("Process", variant="primary")

        process_button.click(
            process_file,
            inputs=[
                file_input, output_dir, format_input, visualize, device,
                td_cfg, tr_cfg, lp_cfg, tsr_cfg, ignore_line_break, figure,
                figure_letter, figure_width, figure_dir
            ],
            outputs=output
        )

    demo.launch()


if __name__ == "__main__":
    main()
