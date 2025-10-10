import argparse
import asyncio
import base64
import os
import re
import sys
import tempfile
import uuid
import warnings
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageChops

from .rstweb_classes import NODE, get_depth, get_left_right
from .rstweb_sql import (
    import_document, get_def_rel, get_max_right, get_multinuc_children_lr,
    get_multinuc_children_lr_ids, get_rst_doc, get_rst_rels, setup_db)

JS_GET_DOCUMENT_HEIGHT = """
let docHeight = Math.max(
  document.body.scrollHeight, document.documentElement.scrollHeight,
  document.body.offsetHeight, document.documentElement.offsetHeight,
  document.body.clientHeight, document.documentElement.clientHeight
);

// we increase the height by 20% because the calculated value is still too small
return Math.round(docHeight * 1.2);
"""

JS_GET_DOCUMENT_WIDTH = """
let docWidth = Math.max(
  document.body.scrollWidth, document.documentElement.scrollWidth,
  document.body.offsetWidth, document.documentElement.offsetWidth,
  document.body.clientWidth, document.documentElement.clientWidth
);

// we increase the width by 3% because the calculated value is still too small
return Math.round(docWidth * 1.03);
"""

PACKAGE_ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT_DIR = PACKAGE_ROOT_DIR / "data"


def _html_to_fragment(full_html: str) -> str:
    """
    Convert a full HTML document into a safe inline fragment:
    - keep <style> and <script> blocks from <head>
    - drop <meta>, <title>, etc.
    - include only the inner HTML of <body>
    """
    # scripts/styles from anywhere (mostly head)
    head_assets = re.findall(
        r"(?is)<style[^>]*>.*?</style>|<script[^>]*>.*?</script>",
        full_html,
    )
    assets_html = "".join(head_assets)

    # body inner HTML (fallback to whole string if no <body>)
    m_body = re.search(r"(?is)<body[^>]*>(.*?)</body>", full_html)
    body_inner = m_body.group(1) if m_body else full_html

    return assets_html + body_inner


def rs3tohtml(rs3_filepath, user='temp_user', project='rstviewer_temp'):
    setup_db()
    import_document(filename=rs3_filepath, project=project, user=user)

    ###GRAPHICAL PARAMETERS###
    top_spacing = 0
    layer_spacing = 60

    templatedir = os.path.join(DATA_ROOT_DIR, 'templates')
    current_doc = os.path.basename(rs3_filepath)
    current_project = project

    with open(os.path.join(templatedir, 'main.html'), 'r', encoding='utf-8') as template:
        header = template.read()

    header = header.replace("**page_title**", "RST Viewer")
    header = header.replace("**doc**", current_doc)

    def _load_asset_text(*path_parts):
        asset_path = os.path.join(DATA_ROOT_DIR, *path_parts)
        with open(asset_path, 'r', encoding='utf-8') as asset_file:
            return asset_file.read()

    header = header.replace(
        '<link rel="stylesheet" href="**css_dir**/rst.css" type="text/css" charset="utf-8"/>',
        '<style>\n' + _load_asset_text('css', 'rst.css') + '\n</style>\n'
                                                           '<style>\n'
                                                           '.rst_rel_wrap{display:inline-flex;align-items:center;justify-content:center}'
                                                           '.rst_rel_label{font-size:8pt;font-weight:bold;'
                                                           ' color:red;background-color:rgba(255,255,255,0.85);'
                                                           ' padding:0 2px;border-radius:3px;user-select:none;'
                                                           ' white-space: nowrap; '
                                                           '}'
                                                           '</style>'
    )

    def _inline_script_tag(script_filename):
        script_text = _load_asset_text('script', script_filename)
        script_text = script_text.replace('</script>', '<\\/script>')
        return '<script>\n' + script_text + '\n</script>'

    header = header.replace(
        '<script src="**script_dir**/jquery-1.11.3.min.js"></script>',
        _inline_script_tag('jquery-1.11.3.min.js'),
    )
    header = header.replace(
        '<script src="**script_dir**/jquery-ui.min.js"></script>',
        _inline_script_tag('jquery-ui.min.js'),
    )

    cpout = ""
    cpout += header
    cpout += '''<div>\n'''

    rels = get_rst_rels(current_doc, current_project)
    def_multirel = get_def_rel("multinuc", current_doc, current_project)
    def_rstrel = get_def_rel("rst", current_doc, current_project)
    multi_options = ""
    rst_options = ""
    rel_kinds = {}
    for rel in rels:
        if rel[1] == "multinuc":
            multi_options += "<option value='" + rel[0] + "'>" + rel[0].replace("_m", "") + '</option>'
            rel_kinds[rel[0]] = "multinuc"
        else:
            rst_options += "<option value='" + rel[0] + "'>" + rel[0].replace("_r", "") + '</option>'
            rel_kinds[rel[0]] = "rst"
    multi_options += "<option value='" + def_rstrel + "'>(satellite...)</option>"

    nodes = {}
    rows = get_rst_doc(current_doc, current_project, user)
    for row in rows:
        if row[7] in rel_kinds:
            relkind = rel_kinds[row[7]]
        else:
            relkind = "span"
        if row[5] == "edu":
            nodes[row[0]] = NODE(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], relkind)
        else:
            nodes[row[0]] = NODE(row[0], 0, 0, row[3], row[4], row[5], row[6], row[7], relkind)

    for key in nodes:
        node = nodes[key]
        get_depth(node, node, nodes)

    for key in nodes:
        if nodes[key].kind == "edu":
            get_left_right(key, nodes, 0, 0, rel_kinds)

    # ---- Adaptive horizontal unit to keep coordinates stable on very wide graphs
    # We need max_right before any anchor/pixel calculations.
    max_right = get_max_right(current_doc, current_project, user)
    # Constrain total width roughly <= 100000k px; keep units reasonable.
    px_unit = max(40, min(100, int(100000 / max(1, max_right))))
    edu_inner_w = px_unit - 4  # keeps same 2px margins as before

    anchors = {}
    pix_anchors = {}

    # Calculate anchor points for nodes (proportional within the parent)
    for key in sorted(nodes, key=lambda id: nodes[id].depth, reverse=True):
        node = nodes[key]
        if node.kind == "edu":
            anchors[node.id] = "0.5"
        if node.parent != "0":
            parent = nodes[node.parent]
            parent_wid = (parent.right - parent.left + 1) * px_unit - 4
            child_wid = (node.right - node.left + 1) * px_unit - 4
            if node.relname == "span":
                if node.id in anchors:
                    anchors[parent.id] = str(
                        ((node.left - parent.left) * px_unit) / parent_wid + float(anchors[node.id]) * float(
                            child_wid / parent_wid))
                else:
                    anchors[parent.id] = str(
                        ((node.left - parent.left) * px_unit) / parent_wid + (0.5 * child_wid) / parent_wid)
            elif node.relkind == "multinuc" and parent.kind == "multinuc":
                lr = get_multinuc_children_lr(node.parent, current_doc, current_project, user)
                lr_wid = (lr[0] + lr[1]) / 2
                lr_ids = get_multinuc_children_lr_ids(node.parent, lr[0], lr[1], current_doc, current_project, user)
                left_child = lr_ids[0]
                right_child = lr_ids[1]
                if left_child == right_child:
                    anchors[parent.id] = "0.5"
                else:
                    if left_child in anchors and right_child in anchors:
                        len_left = nodes[left_child].right - nodes[left_child].left + 1
                        len_right = nodes[right_child].right - nodes[right_child].left + 1
                        anchors[parent.id] = str(((float(anchors[left_child]) * len_left * px_unit + float(
                            anchors[right_child]) * len_right * px_unit + (nodes[
                                                                               right_child].left - parent.left) * px_unit) / 2) / parent_wid)
                    else:
                        anchors[parent.id] = str((lr_wid - parent.left + 1) / (parent.right - parent.left + 1))
            else:
                if not parent.id in anchors:
                    anchors[parent.id] = "0.5"

    # Place anchor element to center on proportional position relative to parent
    for key in nodes:
        node = nodes[key]
        pix_anchors[node.id] = str(int(
            3 + node.left * px_unit - px_unit - 39 +
            float(anchors[node.id]) * ((node.right - node.left + 1) * px_unit - 4)
        ))

    for key in nodes:
        node = nodes[key]
        if node.kind != "edu":
            g_wid = str(int((node.right - node.left + 1) * px_unit - 4))
            cpout += (
                    '<div id="lg' + node.id + '" class="group" style="left: '
                    + str(int(node.left * px_unit - px_unit))
                    + 'px; width: '
                    + g_wid
                    + 'px; top:'
                    + str(int(top_spacing + layer_spacing + node.depth * layer_spacing))
                    + 'px; z-index:1">\n'
            )
            cpout += (
                    '\t<div id="wsk'
                    + node.id
                    + '" class="whisker" style="width:'
                    + g_wid
                    + 'px;"></div>\n</div>\n'
            )
            cpout += (
                    '<div id="g' + node.id + '" class="num_cont" style="position: absolute; left:'
                    + pix_anchors[node.id] + 'px; top:' + str(
                int(4 + top_spacing + layer_spacing + node.depth * layer_spacing))
                    + 'px; z-index:' + str(int(200 - (node.right - node.left))) + '">\n'
            )
            cpout += '\t<table class="btn_tb">\n\t\t<tr>'
            cpout += '\n\t\t\t<td rowspan="2"><span class="num_id">' + str(int(node.left)) + "-" + str(
                int(node.right)) + '</span></td>\n'
            cpout += '\t</table>\n</div>\n<br/>\n\n'

        elif node.kind == "edu":
            cpout += (
                    '<div id="edu'
                    + str(node.id)
                    + '" class="edu" title="'
                    + str(node.id)
                    + '" style="left:'
                    + str(int(int(node.id) * px_unit - px_unit))
                    + 'px; top:'
                    + str(int(top_spacing + layer_spacing + node.depth * layer_spacing))
                    + 'px; width: '
                    + str(int(edu_inner_w))
                    + 'px">\n'
            )
            cpout += '\t<div id="wsk' + node.id + '" class="whisker" style="width:' + str(
                int(edu_inner_w)) + 'px;"></div>'
            cpout += '\n\t<div class="edu_num_cont">'
            cpout += '\n\t\t<table class="btn_tb">\n\t\t\t<tr>'
            cpout += '\n\t\t\t\t<td rowspan="2"><span class="num_id">&nbsp;' + str(
                int(node.left)) + '&nbsp;</span></td>\n'
            cpout += '</table>\n</div>' + node.text + '</div>\n'

    jsplumb_src = _load_asset_text('script', 'jquery.jsPlumb-1.7.5-min.js')
    cpout += '<script>\n' + jsplumb_src + '\n</script>\n<script>\n'

    cpout += 'function select_my_rel(options,my_rel){'
    cpout += 'var multi_options = "' + multi_options + '";'
    cpout += 'var rst_options = "' + rst_options + '";'
    cpout += 'if (options =="multi"){options = multi_options;} else {options=rst_options;}'
    cpout += '      return options.replace("<option value=' + "'" + '"' + '+my_rel+' + '"' + "'" + '","<option selected=' + "'" + 'selected' + "'" + ' value=' + "'" + '"' + '+my_rel+' + '"' + "'" + '");'
    cpout += '          }\n'

    cpout += '''function rel_display(rel){
        return (rel || "").replace(/_(m|r)$/, "");
    }
    function make_relchooser(id, option_type, rel){
        var wrap = document.createElement("span");
        wrap.className = "rst_rel_wrap";
        var hidden = document.createElement("input");
        hidden.type = "hidden";
        hidden.id = "sel" + id.replace("n", "");
        hidden.value = rel || "";
        wrap.appendChild(hidden);
        var label = document.createElement("span");
        label.className = "rst_rel_label";
        label.textContent = rel_display(rel);
        wrap.appendChild(label);
        return $(wrap);
    }'''

    cpout += '''
        var rstStyleTarget = document.body || document.documentElement;
        var rstComputedStyle = (rstStyleTarget && window.getComputedStyle) ? window.getComputedStyle(rstStyleTarget) : null;
        var rstConnectorStroke = '';
        var rstEndpointFill = '';
        if (rstComputedStyle){
            var connectorStrokeValue = rstComputedStyle.getPropertyValue('--rst-connector-stroke');
            if (connectorStrokeValue){
                rstConnectorStroke = connectorStrokeValue.replace(/^\s+|\s+$/g, '');
            }
            var endpointFillValue = rstComputedStyle.getPropertyValue('--rst-line-color');
            if (endpointFillValue){
                rstEndpointFill = endpointFillValue.replace(/^\s+|\s+$/g, '');
            }
        }
        if (!rstConnectorStroke){
            rstConnectorStroke = 'rgba(0,0,0,0.5)';
        }
        if (!rstEndpointFill){
            rstEndpointFill = '#000000';
        }
            jsPlumb.importDefaults({
            PaintStyle : {
                lineWidth:2,
                strokeStyle: rstConnectorStroke
            },
            HoverPaintStyle : {
                strokeStyle: rstConnectorStroke
            },
            Endpoints : [ [ "Dot", { radius:1 } ], [ "Dot", { radius:1 } ] ],
              EndpointStyles : [{ fillStyle: rstEndpointFill }, { fillStyle: rstEndpointFill }],
              EndpointHoverStyles : [{ fillStyle: rstEndpointFill }, { fillStyle: rstEndpointFill }],
              Anchor:"Top",
                Connector : [ "Bezier", { curviness:50 } ]
            });
        jsPlumb.bind("connection", function(info){
            var overlays = info.connection.getOverlays();
            for (var overlayId in overlays){
                if (!overlays.hasOwnProperty(overlayId)){
                    continue;
                }
                var overlay = overlays[overlayId];
                if (overlay && overlay.type === "Arrow" && overlay.setPaintStyle){
                    overlay.setPaintStyle({ strokeStyle: rstConnectorStroke, fillStyle: rstConnectorStroke });
                }
            }
        });
             jsPlumb.ready(function() {

    jsPlumb.setContainer(document.getElementById("inner_canvas"));
    '''

    cpout += "jsPlumb.setSuspendDrawing(true);"

    for key in nodes:
        node = nodes[key]
        if node.kind == "edu":
            node_id_str = "edu" + node.id
        else:
            node_id_str = "g" + node.id
        cpout += 'jsPlumb.makeSource("' + node_id_str + '", {anchor: "Top", filter: ".num_id", allowLoopback:false});'
        cpout += 'jsPlumb.makeTarget("' + node_id_str + '", {anchor: "Top", filter: ".num_id", allowLoopback:false});'

    # Connect nodes
    for key in nodes:
        node = nodes[key]
        if node.parent != "0":
            parent = nodes[node.parent]
            if node.kind == "edu":
                node_id_str = "edu" + node.id
            else:
                node_id_str = "g" + node.id
            if parent.kind == "edu":
                parent_id_str = "edu" + parent.id
            else:
                parent_id_str = "g" + parent.id

            if node.relname == "span":
                cpout += 'jsPlumb.connect({source:"' + node_id_str + '",target:"' + parent_id_str + '", connector:"Straight", anchors: ["Top","Bottom"]});'
            elif parent.kind == "multinuc" and node.relkind == "multinuc":
                cpout += 'jsPlumb.connect({source:"' + node_id_str + '",target:"' + parent_id_str + '", connector:"Straight", anchors: ["Top","Bottom"], overlays: [ ["Custom", {create:function(component) {return make_relchooser("' + node.id + '","multi","' + node.relname + '");},location:0.2,id:"customOverlay"}]]});'
            else:
                cpout += 'jsPlumb.connect({source:"' + node_id_str + '",target:"' + parent_id_str + '", overlays: [ ["Arrow" , { width:12, length:12, location:0.95 }],["Custom", {create:function(component) {return make_relchooser("' + node.id + '","rst","' + node.relname + '");},location:0.1,id:"customOverlay"}]]});'

    cpout += '''
        jsPlumb.setSuspendDrawing(false,true);

        jsPlumb.bind("connection", function(info) {
           source = info.sourceId.replace(/edu|g/,"")
           target = info.targetId.replace(/edu|g/g,"")
        });

        jsPlumb.bind("beforeDrop", function(info) {
            $(".minibtn").prop("disabled",true);
    '''

    cpout += '''
            var node_id = "n"+info.sourceId.replace(/edu|g|lg/,"");
            var new_parent_id = "n"+info.targetId.replace(/edu|g|lg/,"");

            nodes = parse_data();
            new_parent = nodes[new_parent_id];
            relname = nodes[node_id].relname;
            new_parent_kind = new_parent.kind;
            if (nodes[node_id].parent != "n0"){
                old_parent_kind = nodes[nodes[node_id].parent].kind;
            }
            else
            {
                old_parent_kind ="none";
            }

            if (info.sourceId != info.targetId){
                if (!(is_ancestor(new_parent_id,node_id))){
                    jsPlumb.select({source:info.sourceId}).detach();
                    if (new_parent_kind == "multinuc"){
                        relname = get_multirel(new_parent_id,node_id,nodes);
                        jsPlumb.connect({source:info.sourceId, target:info.targetId, connector:"Straight", anchors: ["Top","Bottom"], overlays: [ ["Custom", {create:function(component) {return make_relchooser(node_id,"multi",relname);},location:0.2,id:"customOverlay"}]]});
                    }
                    else{
                        jsPlumb.connect({source:info.sourceId, target:info.targetId, overlays: [ ["Arrow" , { width:12, length:12, location:0.95 }],["Custom", {create:function(component) {return make_relchooser(node_id,"rst",relname);},location:0.1,id:"customOverlay"}]]});
                    }
                    new_rel = document.getElementById("sel"+ node_id.replace("n","")).value;
                    act('up:' + node_id.replace("n","") + ',' + new_parent_id.replace("n",""));
                    update_rel(node_id,new_rel,nodes);
                    recalculate_depth(parse_data());
                }
            }

            $(".minibtn").prop("disabled",false);

        });

    });
</script>

</div>
</body>
</html>
'''
    return cpout


def rs3topng(
        rs3_filepath: Union[str, os.PathLike],
        png_filepath: Optional[Union[str, os.PathLike]] = None,
        base64_encoded: bool = False,
        *,
        device_scale_factor: int = 2,
        timeout_ms: int = 10_000,
):
    """
    Convert an RS3 file into a PNG image of the RST tree using Playwright/Chromium.

    Parameters
    ----------
    rs3_filepath : str | PathLike
        Path to the .rs3 file.
    png_filepath : str | PathLike | None
        If provided, write the PNG (or base64 text) to this path and return None.
        If None, return the PNG bytes or base64 string.
    base64_encoded : bool
        When True, produce/return a base64-encoded PNG instead of raw bytes.
    device_scale_factor : int
        Effective pixel ratio (2 = retina-like output).
    timeout_ms : int
        Max time (ms) to wait for layout scripts to finish.

    Returns
    -------
    bytes | str | None
        - bytes → PNG bytes (if no output path and base64_encoded=False)
        - str → base64 PNG (if base64_encoded=True)
        - None → when written to a file
    """
    try:
        # Detect if we're in an async environment (e.g., Jupyter)
        asyncio.get_running_loop()
        raise RuntimeError(
            "Detected running asyncio loop. "
            "Use the async-safe version instead:\n"
            "    await rs3topng_async(...)\n"
        )
    except RuntimeError:
        # No running loop — safe to use sync Playwright
        pass

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    except Exception as e:
        raise ImportError(
            "Playwright not installed.\n"
            "Run:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        ) from e

    html_str = rs3tohtml(os.fspath(rs3_filepath))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            device_scale_factor=device_scale_factor,
            color_scheme="light",
        )
        page = context.new_page()

        try:
            page.set_content(html_str, wait_until="load", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            # Continue even if scripts load slowly
            pass

        page.wait_for_timeout(50)  # let jsPlumb/layout settle

        doc_width = int(page.evaluate(JS_GET_DOCUMENT_WIDTH))
        doc_height = int(page.evaluate(JS_GET_DOCUMENT_HEIGHT))
        doc_width = max(doc_width, 320)
        doc_height = max(doc_height, 240)

        page.set_viewport_size({"width": doc_width, "height": doc_height})

        png_bytes: bytes = page.screenshot(full_page=True, type="png")

        context.close()
        browser.close()

    # Return or save
    if base64_encoded:
        png_text = base64.b64encode(png_bytes).decode("ascii")
        if png_filepath:
            Path(png_filepath).write_text(png_text, encoding="utf-8")
            return None
        return png_text

    if png_filepath:
        Path(png_filepath).write_bytes(png_bytes)
        return None

    return png_bytes


async def rs3topng_async(
        rs3_filepath,
        png_filepath=None,
        base64_encoded: bool = False,
        *,
        device_scale_factor: int = 2,
        viewport_width: int = 1600,
        viewport_height: int = 1000,
        timeout_ms: int = 10_000,
        margin_px: int = 12,  # extra padding around the graph
):
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

    def trim_whitespace(png_bytes: bytes, pad: int = 12) -> bytes:
        im = Image.open(BytesIO(png_bytes)).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()
        if not bbox:
            return png_bytes  # nothing to trim
        # expand bbox by padding and clip to image bounds
        left = max(bbox[0] - pad, 0)
        top = max(bbox[1] - pad, 0)
        right = min(bbox[2] + pad, im.width)
        bottom = min(bbox[3] + pad, im.height)
        cropped = im.crop((left, top, right, bottom))

        out = BytesIO()
        cropped.save(out, format="PNG")
        return out.getvalue()

    html_str = rs3tohtml(os.fspath(rs3_filepath))

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            device_scale_factor=device_scale_factor,
            viewport={"width": viewport_width, "height": viewport_height},
            color_scheme="light",
        )
        page = await context.new_page()
        try:
            await page.set_content(html_str, wait_until="load", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            pass

        # Let jsPlumb/jQuery settle
        await page.wait_for_timeout(100)

        # Compute a tight bounding box of everything inside #inner_canvas
        # (EDUs, groups, and jsPlumb SVGs)
        bbox = await page.evaluate(r"""
(() => {
  const root = document.querySelector('#inner_canvas') || document.body;
  const items = root.querySelectorAll(
    '.edu, .group, .num_cont, svg, canvas, path, ._jsPlumb_connector, ._jsPlumb_endpoint'
  );

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const push = (r) => {
    if (!r) return;
    minX = Math.min(minX, r.left);
    minY = Math.min(minY, r.top);
    maxX = Math.max(maxX, r.right);
    maxY = Math.max(maxY, r.bottom);
  };

  // include root rect too (some themes draw lines on it)
  push(root.getBoundingClientRect());
  items.forEach(el => {
    const r = el.getBoundingClientRect?.();
    if (r && Number.isFinite(r.width) && Number.isFinite(r.height)) push(r);
  });

  if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
    // Fallback: whole document
    const doc = document.documentElement;
    return {x: 0, y: 0, width: doc.scrollWidth, height: doc.scrollHeight};
  }
  return {x: Math.floor(minX), y: Math.floor(minY),
          width: Math.ceil(maxX - minX), height: Math.ceil(maxY - minY)};
})()
        """)

        # Add a little margin
        x = max(bbox["x"] - margin_px, 0)
        y = max(bbox["y"] - margin_px, 0)
        w = bbox["width"] + margin_px * 2
        h = bbox["height"] + margin_px * 2

        # Ensure the viewport can cover the clip area
        await page.set_viewport_size({
            "width": max(viewport_width, x + w + 20),
            "height": max(viewport_height, y + h + 20),
        })

        png_bytes: bytes = await page.screenshot(
            type="png",
            clip={"x": x, "y": y, "width": w, "height": h},
        )

        await context.close()
        await browser.close()

    png_bytes = trim_whitespace(png_bytes)

    if base64_encoded:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        if png_filepath:
            Path(png_filepath).write_text(b64, encoding="utf-8")
            return None
        return b64

    if png_filepath:
        Path(png_filepath).write_bytes(png_bytes)
        return None

    return png_bytes


async def rs3topdf_async(
        rs3_filepath,
        pdf_path: str,
        *,
        device_scale_factor: int = 2,
        viewport_width: int = 1600,
        viewport_height: int = 1000,
        timeout_ms: int = 10_000,
        margin_px: int = 12,  # extra padding around the graph
):
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

    html_str = rs3tohtml(os.fspath(rs3_filepath))

    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
        except Exception as e:
            raise ImportError(
                "Browser is not installed.\n"
                "Run:\n"
                "  playwright install chromium"
            ) from e

        context = await browser.new_context(
            device_scale_factor=device_scale_factor,
            viewport={"width": viewport_width, "height": viewport_height},
            color_scheme="light",
        )
        page = await context.new_page()
        try:
            await page.set_content(html_str, wait_until="load", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            pass
        await page.wait_for_timeout(100)

        # Find tight bbox of the graph area
        bbox = await page.evaluate(r"""
(() => {
  const root = document.querySelector('#inner_canvas') || document.body;
  const items = root.querySelectorAll('.edu, .group, .num_cont, svg, canvas, path, ._jsPlumb_connector, ._jsPlumb_endpoint');
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  const push = r => { if(!r) return; minX=Math.min(minX,r.left); minY=Math.min(minY,r.top); maxX=Math.max(maxX,r.right); maxY=Math.max(maxY,r.bottom); };
  push(root.getBoundingClientRect());
  items.forEach(el => { const r=el.getBoundingClientRect?.(); if(r && isFinite(r.width) && isFinite(r.height)) push(r); });
  if (!isFinite(minX)) {
    const doc = document.documentElement;
    return {x:0,y:0,width:doc.scrollWidth,height:doc.scrollHeight};
  }
  return {x:Math.floor(minX),y:Math.floor(minY),width:Math.ceil(maxX-minX),height:Math.ceil(maxY-minY)};
})()
        """)

        # Expand by margin and normalize
        x = max(bbox["x"] - margin_px, 0)
        y = max(bbox["y"] - margin_px, 0)
        w = bbox["width"] + margin_px * 2
        h = bbox["height"] + margin_px * 2

        # Resize the page’s layout to exactly the bbox, then print to PDF
        await page.add_style_tag(content=f"""
html, body {{
  margin: 0 !important;
  padding: 0 !important;
  width: {w}px !important;
  height: {h}px !important;
  overflow: hidden !important;
}}
#inner_canvas {{
  position: absolute !important;
  left: {-x}px !important;
  top: {-y}px !important;
}}
@page {{
  size: {w}px {h}px;
  margin: 0;
}}
""")

        # Chromium's PDF is vector (text + paths)
        await page.pdf(
            path=pdf_path,
            width=f"{w}px",
            height=f"{h}px",
            print_background=True,
            prefer_css_page_size=True,
        )

        await context.close()
        await browser.close()


class RenderedRST(str):
    """String subclass that cooperates with IPython display hooks."""

    def __new__(cls, value, *, already_displayed, display_override=None):
        rendered = super().__new__(cls, value)
        rendered._already_displayed = already_displayed
        rendered._display_override = display_override
        return rendered

    def _repr_html_(self):
        if getattr(self, "_already_displayed", False):
            return ""
        return getattr(self, "_display_override", None) or str(self)


def _new_root_id():
    return "rst-root-" + uuid.uuid4().hex


def _wrap_for_colab(html_str):
    # Use fragment so we don't inject <html>/<head>/<body> inside a <div>
    frag = _html_to_fragment(html_str)
    root_id = _new_root_id()
    return (
        f'<div id="{root_id}" style="margin:0;padding:0;">{frag}</div>\n'
        "<script>\n"
        "(function() {\n"
        "  const maxFrames = 1000;\n"
        "  const stableNeeded = 20;\n"
        "  let last = -1;\n"
        "  let stable = 0;\n"
        "  let frames = 0;\n"
        "  function hDoc() {\n"
        "    return Math.max(\n"
        "      document.body.scrollHeight,\n"
        "      document.documentElement.scrollHeight,\n"
        "      document.body.offsetHeight,\n"
        "      document.documentElement.offsetHeight,\n"
        "      document.body.clientHeight,\n"
        "      document.documentElement.clientHeight\n"
        "    );\n"
        "  }\n"
        "  function tick() {\n"
        "    try {\n"
        "      const h = hDoc();\n"
        "      if (h !== last) {\n"
        "        last = h;\n"
        "        stable = 0;\n"
        "        google.colab.output.setIframeHeight(h, false);\n"
        "      } else if (++stable === stableNeeded) {\n"
        "        google.colab.output.setIframeHeight(last, false);\n"
        "        return;\n"
        "      }\n"
        "    } catch (e) {}\n"
        "    if (++frames < maxFrames) requestAnimationFrame(tick);\n"
        "    else { try { google.colab.output.setIframeHeight(hDoc(), false); } catch(e) {} }\n"
        "  }\n"
        "  setTimeout(() => requestAnimationFrame(tick), 0);\n"
        "  setTimeout(() => { try { google.colab.output.setIframeHeight(hDoc(), false); } catch(e) {} }, 500);\n"
        "  setTimeout(() => { try { google.colab.output.setIframeHeight(hDoc(), false); } catch(e) {} }, 1500);\n"
        "  setTimeout(() => { try { google.colab.output.setIframeHeight(hDoc(), false); } catch(e) {} }, 3000);\n"
        "})();\n"
        "</script>"
    )


def _wrap_for_notebook(html_str):
    # Use fragment so we don't inject <html>/<head>/<body> inside a <div>
    frag = _html_to_fragment(html_str)
    root_id = _new_root_id()

    return (
        f'<div id="{root_id}" '
        'style="margin:0;padding:0;max-width:100%;overflow-x:auto;overflow-y:visible;">'
        f"{frag}</div>\n"
        f"<script data-rst-resize=\"{root_id}\">\n"
        "(function() {\n"
        f"  var ROOT_ID = {root_id!r};\n"
        "  var cachedScript = null;\n"
        "  function matches(el, sel){var p=Element.prototype;var f=p.matches||p.msMatchesSelector||p.webkitMatchesSelector;return el&&f?f.call(el,sel):false}\n"
        "  function closest(el, sel){if(!el)return null;if(el.closest)return el.closest(sel);while(el&&el.nodeType===1){if(matches(el,sel))return el;el=el.parentElement}return null}\n"
        "  function getScript(){var s=document.currentScript;if(s&&s.dataset&&s.dataset.rstResize===ROOT_ID){cachedScript=s;return s}if(!cachedScript||!cachedScript.isConnected){cachedScript=document.querySelector('script[data-rst-resize=\"'+ROOT_ID+'\"]')}return cachedScript}\n"
        "  function getRoot(s){if(!s)return null;var r=s.previousElementSibling;if(r&&r.id===ROOT_ID)return r;if(s.parentElement){r=s.parentElement.querySelector('#'+ROOT_ID);if(r)return r}return document.getElementById(ROOT_ID)}\n"
        "  function styleEl(el){if(!el)return; if(el.classList&&el.classList.contains('output_scroll'))el.classList.remove('output_scroll'); el.style.maxHeight='none'; el.style.height='auto'; el.style.minHeight='0'; el.style.overflow=''; el.style.overflowX='auto'; el.style.overflowY='visible'}\n"
        "  function measure(root){if(!root)return 0;var rect=root.getBoundingClientRect();var baseTop=rect?rect.top:0;var maxBottom=rect?rect.bottom:0;var els=root.getElementsByTagName('*');for(var i=0;i<els.length;i++){var e=els[i];if(!e||!e.getBoundingClientRect)continue;var r=e.getBoundingClientRect();if(r&&typeof r.bottom==='number'&&r.bottom>maxBottom)maxBottom=r.bottom}var computed=[root.scrollHeight||0,root.offsetHeight||0,rect?rect.height:0,maxBottom-baseTop];var h=0;for(var j=0;j<computed.length;j++){if(computed[j]>h)h=computed[j]}return Math.ceil(h)}\n"
        "  function apply(el,h){if(!el||!h)return;el.style.minHeight=h+'px';el.style.height=h+'px'}\n"
        "  function adjust(){var s=getScript();if(!s)return;var root=getRoot(s);if(!root)return;var container=closest(root,'.output_subarea')||closest(root,'.jp-RenderedHTMLCommon')||closest(root,'.jp-OutputArea-output')||root.parentElement;var scrollable=container?container.querySelector('.output_scroll'):null;var direct=[];if(scrollable)direct.push(scrollable);if(container)direct.push(container);var wrappers=[];var p=container?container.parentElement:null;while(p){wrappers.push(p);if(matches(p,'.output_area')||matches(p,'.jp-OutputArea'))break;p=p.parentElement}var h=measure(root);if(h)h=h+1;direct.forEach(function(el){styleEl(el);apply(el,h)});wrappers.forEach(function(el){styleEl(el);apply(el,h)});if(root)root.style.minHeight=h?h+'px':''}\n"
        "  function run(){adjust();requestAnimationFrame(adjust);setTimeout(adjust,0);setTimeout(adjust,250)}\n"
        "  run();window.addEventListener('resize', run);\n"
        "})();\n"
        "</script>"
    )


def render(rs3_source, *, display_inline=True, colab=False):
    """Render an RST tree and optionally display it inline.

    Parameters
    ----------
    rs3_source : str or os.PathLike or IO[str]
        Either the textual content of an ``.rs3`` file, a path to an
        ``.rs3`` file, or a text IO object containing the file contents.
    display_inline : bool, optional
        When ``True`` (the default) the resulting HTML is displayed using
        :mod:`IPython.display` when available. Regardless of the value, the
        HTML string is returned to the caller.
    colab : bool, optional
        When ``True`` the returned HTML is wrapped with JavaScript that keeps
        the Google Colab output cell height synchronized with the rendered
        document. The wrapped HTML is also used for inline display.

    Returns
    -------
    RenderedRST
        The rendered HTML representation of the RST tree.
    """

    temp_path = None

    if hasattr(rs3_source, "read"):
        rs3_content = rs3_source.read()
        if isinstance(rs3_content, bytes):
            rs3_content = rs3_content.decode('utf8')
        temp_path = tempfile.NamedTemporaryFile(suffix='.rs3', delete=False)
        try:
            temp_path.write(rs3_content.encode('utf8'))
        finally:
            temp_path.close()
        rs3_path = temp_path.name
    elif isinstance(rs3_source, (os.PathLike, str)) and os.path.exists(rs3_source):
        rs3_path = os.fspath(rs3_source)
    else:
        if isinstance(rs3_source, bytes):
            rs3_content = rs3_source.decode('utf8')
        else:
            rs3_content = str(rs3_source)
        temp_path = tempfile.NamedTemporaryFile(suffix='.rs3', delete=False)
        try:
            temp_path.write(rs3_content.encode('utf8'))
        finally:
            temp_path.close()
        rs3_path = temp_path.name

    try:
        html_str = rs3tohtml(rs3_path)
    finally:
        if temp_path is not None:
            os.unlink(temp_path.name)

    already_displayed = False
    if colab:
        display_html = _wrap_for_colab(html_str)
    else:
        display_html = _wrap_for_notebook(html_str)
    if display_inline:
        try:
            from IPython.display import HTML, display
        except ImportError:
            warnings.warn(
                'IPython is not available; returning HTML string without displaying it.',
                RuntimeWarning)
        else:
            display(HTML(display_html))
            already_displayed = True

    return RenderedRST(
        html_str,
        already_displayed=already_displayed,
        display_override=display_html if display_html != html_str else None)


def cli(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Convert an RS3 file into an HTML file containing the RST tree.")
    parser.add_argument('rs3_file')
    parser.add_argument('output_file', nargs='?')
    parser.add_argument(
        '-f', '--output-format', nargs='?', default='html',
        help="output format: html (default), png, png-base64")
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help="run the program in pudb")

    args = parser.parse_args(argv)

    if args.debug:
        import pudb;
        pudb.set_trace()

    if args.output_format == 'png':
        if args.output_file:
            rs3topng(args.rs3_file, args.output_file)
            sys.exit(0)
        else:
            sys.stderr.write("No PNG output file given.\n")
            sys.exit(1)
    elif args.output_format == 'png-base64':
        if args.output_file:
            rs3topng(args.rs3_file, args.output_file, base64_encoded=True)
            sys.exit(0)
        else:
            base64_png_str = rs3topng(args.rs3_file, base64_encoded=True)
            sys.stdout.write(base64_png_str)
            sys.exit(0)

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf8') as outfile:
            outfile.write(rs3tohtml(args.rs3_file))
    else:
        sys.stdout.write(rs3tohtml(args.rs3_file))


if __name__ == '__main__':
    cli(sys.argv[1:])
