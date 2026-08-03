# -*- coding: utf-8 -*-
"""
webapp/capture_figures.py — captures the interface figures at publication resolution.

WHY A SCRIPT AND NOT A SCREENSHOT KEY. The resolution of a screen capture is not a
setting you apply afterwards: it is pixels divided by printed width. Resampling a
1431 px capture up to 4252 px invents interpolated pixels and the typesetter sees the
softness. The only way to gain real pixels is to RENDER at a higher device pixel
ratio, so the browser rasterises text and vector chrome natively at that scale. That
is what `device_scale_factor` does here, and it is why this exists as a script rather
than as an instruction to press a key.

WHAT THE TARGET IS. Wiley, "Guidelines for the Preparation of Figures" (1 Sep 2016),
linked from the Cytometry Part A author guidelines:

  · "If a figure includes both line art and images, follow the line art guidelines."
    A UI capture is text plus rendered images, so line art applies — 600 dpi, not the
    300 dpi that photographs get.
  · line art: 600 dpi for peer review, 600-1000 dpi post-acceptance
  · width: 80 mm (quarter page) or 180 mm (half to full page), and in BOTH cases
    "1800px minimum"
  · under 10 MB per file
  · "name figure files only with the word 'figure' and the appropriate number"

The arithmetic this script is built on:

    viewport 1440 CSS px  x  device_scale_factor 3  =  4320 device px
    4320 px / 7.087 in (180 mm)                     =  610 dpi        -> clears 600

A scale factor of 2 gives 406 dpi: past the 1800 px floor and past 300, short of 600.

THE pHYs CHUNK IS NOT OPTIONAL. A PNG carrying no physical-size metadata is assumed
to be 96 dpi, so Word would place a 4320 px image 45 inches wide and then compress it
back down. Writing pHYs at 600 dpi makes the placed width 7.2 in = 183 mm on its own.

AND DO NOT PASTE THE RESULT INTO THE .docx. Word's default image handling is
"Print (220 ppi)", which is exactly what every figure currently embedded in the
manuscript was reduced to — measured, all six at 220 ppi. Supply the files
separately, which the author guidelines ask for anyway.

WHICH BROWSER. `--browser chromium` (the default) uses the build Playwright downloads
and pins itself. It is not Google Chrome, it does not touch an installed browser, and
because the version is pinned the figure can be regenerated identically later — which
is the whole point of producing it from a script. `--browser msedge` drives the
Microsoft Edge already on the machine instead; Edge is Chromium, so the rendering is
the same, but Edge updates itself and the exact build is no longer under our control.

    pip install playwright
    python -m playwright install chromium      # only for --browser chromium

    python webapp/capture_figures.py --out figures
    python webapp/capture_figures.py --out figures --browser msedge
    python webapp/capture_figures.py --out figures --theme light --scale 4
"""
import argparse
import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── the numbers from the Wiley guidelines ────────────────────────────────────
LARGURA_GRANDE_MM = 180.0        # half to full page
LARGURA_PEQUENA_MM = 80.0        # quarter page
PISO_PX = 1800                   # stated minimum for BOTH canvas sizes
ALVO_DPI = 600                   # line art, peer review
LIMITE_MB = 10.0

MM_POR_POL = 25.4
POL_GRANDE = LARGURA_GRANDE_MM / MM_POR_POL      # 7.0866
POL_PEQUENA = LARGURA_PEQUENA_MM / MM_POR_POL    # 3.1496

# The app renders directly at the .hf.space origin. Going through
# huggingface.co/spaces/... wraps it in an iframe plus the platform header, which
# lands in the capture. The current Figure 2 and Figure 7 do show that header; keep
# --chrome if you want to preserve that look, since the caption does say the tool is
# "hosted on the Hugging Face platform".
APP_DIRETO = "https://nmariotto-scratch-assay-segmentation.hf.space"
APP_COM_CHROME = "https://huggingface.co/spaces/nmariotto/Scratch-assay-segmentation"


def relata(caminho, alvo_pol=POL_GRANDE, rotulo="180 mm"):
    """Measures what was actually produced against the guideline floors."""
    from PIL import Image
    im = Image.open(caminho)
    w, h = im.size
    mb = os.path.getsize(caminho) / 2 ** 20
    dpi = w / alvo_pol
    ok_piso = w >= PISO_PX
    ok_dpi = dpi >= ALVO_DPI
    ok_mb = mb <= LIMITE_MB
    print(f"  {os.path.basename(caminho)}")
    print(f"    {w} x {h} px · {mb:.1f} MB · pHYs {im.info.get('dpi')}")
    print(f"    at {rotulo}: {dpi:.0f} dpi")
    print(f"    floor {PISO_PX} px  {'ok' if ok_piso else 'FAIL'}   "
          f"line art {ALVO_DPI} dpi  {'ok' if ok_dpi else 'FAIL'}   "
          f"under {LIMITE_MB:.0f} MB  {'ok' if ok_mb else 'FAIL'}")
    return ok_piso and ok_dpi and ok_mb


def apara(im, margem_px=48, corta_lateral=True):
    """Frames the figure on the content: drops the sidebar and the empty margins.

    Three things are removed, all of them dead figure area. Scaled to 180 mm, every
    pixel spent on them is a pixel not spent on what the caption points at.

      · the left sidebar, which holds only the About/Citation expander and none of
        the workflow the caption describes;
      · the blank strip below the last widget, left over from a viewport made tall
        enough to hold the expanded panel;
      · the blank strip above the title.

    This is a crop, not a re-layout. Hiding the sidebar with CSS would let the main
    column reflow and widen, which changes the proportions a user actually sees;
    cropping keeps the rendering exactly as served and only chooses the frame.

    Finding the sidebar edge needs care: Streamlit paints it in its own shade and runs
    it the full height, so a naive row scan finds "content" on every row and trims
    nothing — which is what the first version of this did. The edge is found by
    walking the bottom row inward until the colour becomes the main background.
    Background is read from a corner rather than assumed, so this holds in either
    theme.
    """
    import numpy as np
    a = np.asarray(im.convert("RGB")).astype(int)
    alt, larg = a.shape[:2]
    fundo = a[-1, -1]

    ultima = a[-1]                                   # bottom row: sidebar, then main
    igual = np.abs(ultima - fundo).sum(1) <= 12
    x0 = 0
    if not igual[0]:                                 # a sidebar is present
        virada = np.where(igual)[0]
        if virada.size:
            x0 = int(virada[0])

    corpo = a[:, x0:]
    linhas = np.where(np.any(np.abs(corpo - fundo).sum(2) > 12, axis=1))[0]
    if linhas.size == 0:
        return im
    topo = max(0, int(linhas[0]) - margem_px)
    base = min(alt, int(linhas[-1]) + margem_px)
    esq = x0 if corta_lateral else 0
    return im.crop((esq, topo, larg, base))


def grava_com_dpi(png_bytes, destino, dpi=ALVO_DPI, cortar=True, corte_topo=0):
    """Writes the PNG with the pHYs chunk set, so Word places it at the right size."""
    from PIL import Image
    im = Image.open(io.BytesIO(png_bytes))
    if corte_topo:
        im = im.crop((0, corte_topo, im.size[0], im.size[1]))
    if cortar:
        im = apara(im)
    im.save(destino, "PNG", dpi=(dpi, dpi), optimize=True)


def espera_streamlit(page, timeout=120000):
    """Streamlit paints in stages; waiting on load alone captures a half-drawn app."""
    page.wait_for_load_state("networkidle", timeout=timeout)
    # the app title is the last thing to settle on first paint
    page.wait_for_selector("text=Scratch Assay Segmentation Tool", timeout=timeout)
    # Streamlit shows a running indicator while a rerun is in flight
    try:
        page.wait_for_selector("[data-testid='stStatusWidget']",
                               state="detached", timeout=15000)
    except Exception:
        pass
    page.wait_for_timeout(2500)


def ajusta_altura(page, largura, margem=40):
    """Grows the viewport to the real content height, then reports it.

    `full_page=True` does not work on this app and silently truncates. Streamlit puts
    the page inside [data-testid="stMain"], which is `overflow: auto` at 100vh, so the
    document body never grows past the viewport and Playwright has nothing to expand
    into. The first Figure 7 came out at 1391 CSS px against a real content height of
    2099 — 40% missing, and the part missing was the Export buttons and the feedback
    block, which is to say the part the caption talks about.

    Measuring the scroller and resizing the viewport to match is what actually works.
    """
    alt = page.evaluate("""() => {
        const c = document.querySelector('[data-testid="stMain"]')
               || document.querySelector('[data-testid="stMainBlockContainer"]');
        return c ? c.scrollHeight : document.documentElement.scrollHeight;
    }""")
    page.set_viewport_size({"width": largura, "height": int(alt) + margem})
    page.wait_for_timeout(1200)
    return int(alt)


def prepara(page, url):
    """Loads the app and puts it in the state both figures share.

    Two things come off before anything else, and for the same reason: they are
    Streamlit furniture rather than the tool, and every pixel they hold is a pixel
    the reader does not get once the figure is scaled to 180 mm.
    """
    page.goto(url, wait_until="domcontentloaded", timeout=180000)
    espera_streamlit(page)

    # The kebab menu floats over the top-right corner. While it is there the top
    # margin cannot be trimmed — it counts as content on the very first rows. It is
    # an overlay, so hiding it costs no reflow.
    page.add_style_tag(content="[data-testid='stToolbar'],"
                               "[data-testid='stMainMenu'],"
                               "[data-testid='stStatusWidget']"
                               "{display:none !important;}")
    page.wait_for_timeout(300)

    # Collapse the sidebar, rather than cropping it out afterwards. It holds only the
    # About/Citation expander, none of the workflow the captions describe, and at 300
    # of 1440 CSS px it is 21% of the frame.
    #
    # Collapsing beats cropping on both counts. The main column REFLOWS to the full
    # width, so the content gets 1440 CSS px instead of 1140 — cropping would have
    # thrown those pixels away and dropped a 4320 px capture to ~3420 px, which is
    # 482 dpi at 180 mm and under the line-art floor. And it is a state the interface
    # actually has: any user can collapse it, so nothing is being staged.
    #
    # The chevron only materialises on hover, so a plain click finds an invisible
    # element and times out. Hover first; fall back to a DOM click.
    try:
        page.hover("[data-testid='stSidebar']", timeout=10000)
        page.wait_for_timeout(500)
        page.locator("[data-testid='stBaseButton-headerNoPadding']").first \
            .click(timeout=10000)
    except Exception:
        page.evaluate("() => { const b = document.querySelector("
                      "'[data-testid=\"stBaseButton-headerNoPadding\"]');"
                      " if (b) b.click(); }")
    page.wait_for_timeout(1200)
    larg = page.evaluate("() => { const s = document.querySelector("
                         "'[data-testid=\"stSidebar\"]');"
                         " return s ? Math.round(s.getBoundingClientRect().width) : 0; }")
    print(f"    sidebar {'collapsed' if larg == 0 else f'still {larg} px wide'}")


def imagem_padrao():
    """The frame Figure 7 has always shown, at the resolution it was acquired at.

    The published Figure 7 uses well A2 at 24 h. The example set only carries the
    Roboflow 640x640 export of it, which the browser would then upscale into the
    capture. The dataset holds the same acquisition at 2452x2056, and it sits in the
    held-out test split — so the figure keeps its specimen and gains the resolution,
    and the demonstration is still out-of-sample.
    """
    import csv as _csv
    import glob as _glob
    from PIL import Image
    mapa = os.path.join("data", "mapping_dataset_final_strat.csv")
    if not os.path.isfile(mapa):
        return None
    disco = {os.path.splitext(os.path.basename(p))[0]: p
             for p in _glob.glob(os.path.join("dataset", "images", "test", "*.*"))}
    escolhas = []
    for r in _csv.DictReader(open(mapa, encoding="utf-8-sig")):
        if r.get("excluida") in ("sim", "1", "True"):
            continue
        if r.get("linha_celular") != "HUVEC" or r.get("timepoint_h") != "24":
            continue
        p = disco.get(os.path.splitext(r.get("arquivo_b", ""))[0])
        if not p or Image.open(p).size[0] < 2000:
            continue
        rot = os.path.splitext(p.replace(os.sep + "images" + os.sep,
                                         os.sep + "labels" + os.sep))[0] + ".txt"
        if os.path.isfile(rot) and sum(1 for _ in open(rot)):
            escolhas.append(p)
    if not escolhas:
        return None
    for p in sorted(escolhas):
        if "a2 24h 2" in os.path.basename(p).lower():
            return p
    return sorted(escolhas)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--url", default=None, help="override the app URL")
    ap.add_argument("--chrome", action="store_true",
                    help="capture through the Hugging Face page, keeping the platform "
                         "header as the current figures show it")
    ap.add_argument("--theme", choices=["light", "dark", "as-is"], default="light",
                    help="app.py sets no theme and there is no .streamlit/config.toml, "
                         "so Streamlit follows the viewer's prefers-color-scheme. The "
                         "dark look of the current figures is not 'the tool's "
                         "appearance' — it is the tool seen from an OS in dark mode. "
                         "Light is the default here because a full-bleed dark "
                         "background costs ink, thins white type under dot gain, and "
                         "turns into a grey slab in greyscale.")
    ap.add_argument("--scale", type=int, default=3,
                    help="device pixel ratio. 3 gives 610 dpi at 180 mm; 2 gives 406. "
                         "3 suffices because the sidebar is collapsed rather than "
                         "cropped, so the full width carries content.")
    ap.add_argument("--viewport", type=int, default=1440, help="viewport width in CSS px")
    ap.add_argument("--sample", default=None,
                    help="image to upload for Figure 7. Defaults to the first file in "
                         "examples/images/")
    ap.add_argument("--only", choices=["2", "7"], default=None)
    ap.add_argument("--recorte", choices=["resultado", "completo"], default="resultado",
                    help="Figure 7 only. 'resultado' starts the frame at the Result "
                         "heading, dropping the Input block. Two reasons. It is "
                         "redundant — Figure 2 exists to document those controls — "
                         "and the full page is 5915 px tall against 4320 wide, which "
                         "at 180 mm is 246 mm of height and does not fit a page. "
                         "Cropping to the result brings it to ~176 mm.")
    ap.add_argument("--stamp", metavar="PNG", default=None,
                    help="do not capture anything: take a PNG produced by hand (the "
                         "DevTools 'Capture full size screenshot' does not write pHYs) "
                         "and stamp it at --dpi, then measure it against the guidelines")
    ap.add_argument("--dpi", type=int, default=ALVO_DPI,
                    help=f"dpi to write into pHYs (default {ALVO_DPI})")
    ap.add_argument("--browser", choices=["chromium", "msedge", "chrome"],
                    default="chromium",
                    help="'chromium' uses Playwright's own pinned build (recommended: "
                         "reproducible, and independent of any installed browser). "
                         "'msedge' drives the Microsoft Edge already installed.")
    args = ap.parse_args()

    # ── stamp mode: no browser, just fix the metadata on a hand-made capture ──
    if args.stamp:
        if not os.path.isfile(args.stamp):
            sys.exit(f"not found: {args.stamp}")
        with open(args.stamp, "rb") as f:
            bruto = f.read()
        grava_com_dpi(bruto, args.stamp, dpi=args.dpi)
        print(f"stamped {args.stamp} at {args.dpi} dpi\n")
        relata(args.stamp)
        return

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright missing:\n"
                 "    pip install playwright\n"
                 "    python -m playwright install chromium")

    url = args.url or (APP_COM_CHROME if args.chrome else APP_DIRETO)
    # Two levers, because either one alone can be ignored. `?__theme=` is Streamlit's
    # own switch; `color_scheme` below sets prefers-color-scheme at the browser level,
    # which is what Streamlit falls back to when no theme is configured — and none is.
    if args.theme in ("light", "dark"):
        url += ("&" if "?" in url else "?") + f"__theme={args.theme}"

    amostra = args.sample
    if amostra is None:
        amostra = imagem_padrao()
    if amostra is None:                      # no dataset checked out; fall back
        d = os.path.join("examples", "images")
        if os.path.isdir(d):
            cand = sorted(f for f in os.listdir(d)
                          if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")))
            amostra = os.path.join(d, cand[0]) if cand else None
            if amostra:
                print(f"[note] dataset not found; falling back to the example set, "
                      f"which holds only the 640x640 Roboflow exports — the browser "
                      f"will upscale them into the capture.\n")

    os.makedirs(args.out, exist_ok=True)
    largura_px = args.viewport * args.scale
    print(f"url        {url}")
    print(f"viewport   {args.viewport} CSS px x DPR {args.scale} = {largura_px} device px")
    print(f"projected  {largura_px/POL_GRANDE:.0f} dpi at 180 mm · "
          f"{largura_px/POL_PEQUENA:.0f} dpi at 80 mm\n")
    if largura_px < PISO_PX:
        sys.exit(f"ABORTED: {largura_px} px is below the {PISO_PX} px floor. "
                 f"Raise --scale or --viewport.")

    tudo_ok = True
    with sync_playwright() as pw:
        # Edge and Chrome are Chromium, so they are launched through the same driver;
        # `channel` picks the installed browser instead of the bundled build.
        lancar = {} if args.browser == "chromium" else {"channel": args.browser}
        try:
            navegador = pw.chromium.launch(**lancar)
        except Exception as e:
            sys.exit(f"could not launch '{args.browser}': {type(e).__name__}: {e}\n"
                     f"For the bundled build: python -m playwright install chromium\n"
                     f"For Edge: make sure Microsoft Edge is installed, or run\n"
                     f"          python -m playwright install msedge")
        print(f"browser    {args.browser}"
              f"{' (Playwright pinned build)' if args.browser == 'chromium' else ' (installed on this machine)'}\n")
        # tall enough to hold the expanded Advanced Settings panel with the model list
        # open below it; the empty tail is trimmed when the file is written
        ctx_kw = {"viewport": {"width": args.viewport, "height": 1500},
                  "device_scale_factor": args.scale}
        if args.theme in ("light", "dark"):
            ctx_kw["color_scheme"] = args.theme
        ctx = navegador.new_context(**ctx_kw)
        page = ctx.new_page()

        # ── Figure 2 · the input screen, model dropdown open ──────────────────
        if args.only in (None, "2"):
            print("Figure 2 — input screen with the model list open")
            prepara(page, url)

            # The caption promises three things, so all three have to be on screen:
            # the model list, the confidence threshold, and the upload control. Two
            # of them are hidden by default.
            #
            # 1. Advanced Settings holds the confidence slider and the physical
            #    calibration fields. Collapsed, the figure cannot support the phrase
            #    "adjust the model confidence threshold".
            try:
                exp = page.get_by_text("Advanced Settings", exact=False).first
                exp.click(timeout=20000)
                # wait for the panel's own content, not for a fixed delay
                page.wait_for_selector("text=Model confidence", timeout=20000)
                page.wait_for_selector("text=Physical calibration", timeout=20000)
                page.wait_for_timeout(600)
                print("    Advanced Settings expanded")
            except Exception as e:
                print(f"    [warn] could not expand Advanced Settings "
                      f"({type(e).__name__}) — the confidence slider will be missing")

            # 2. The model list. Done AFTER the panel, so the list is drawn on top
            #    rather than being pushed off by the expansion.
            #
            #    The widget is NOT `[data-baseweb='select']`. This Streamlit build uses
            #    react-aria: an <input role="combobox" aria-autocomplete="list"> inside
            #    [data-testid='stSelectbox']. And the options are rendered into a
            #    PORTAL hanging off <body>, not inside the listbox — so the descendant
            #    selector "[role='listbox'] [role='option']" matches nothing even with
            #    the list open, which is what made the first attempt time out. Plain
            #    [role='option'] is what finds them.
            try:
                sel = page.locator("[role='combobox']").first
                if sel.count() == 0:
                    sel = page.locator("[data-testid='stSelectbox']").first
                sel.scroll_into_view_if_needed(timeout=10000)
                sel.click(timeout=20000)
                page.wait_for_selector("[role='option']", state="visible", timeout=20000)
                page.wait_for_timeout(700)
                n = page.locator("[role='option']").count()
                vis = [page.locator("[role='option']").nth(i).inner_text().strip()
                       for i in range(n)
                       if page.locator("[role='option']").nth(i).is_visible()]
                print(f"    model list open ({len(vis)} visible: {', '.join(vis)})")
            except Exception as e:
                print(f"    [warn] could not open the dropdown ({type(e).__name__}) — "
                      f"the figure will show only the selected model")

            # viewport, not full_page: the open listbox is absolutely positioned and a
            # full-page capture can scroll out from under it. The viewport is set tall
            # above and the empty tail is trimmed on write.
            destino = os.path.join(args.out, "figure2.png")
            grava_com_dpi(page.screenshot(full_page=False), destino)
            tudo_ok &= relata(destino)
            print()

        # ── Figure 7 · a complete result ──────────────────────────────────────
        if args.only in (None, "7"):
            print("Figure 7 — a full segmentation result")
            if not amostra or not os.path.isfile(amostra):
                print("    [skip] no sample image. Pass --sample <file>.")
            else:
                from PIL import Image as _Im
                print(f"    uploading {os.path.basename(amostra)} "
                      f"({'x'.join(map(str, _Im.open(amostra).size))})")
                prepara(page, url)
                # Advanced Settings stays collapsed here on purpose. This caption is
                # about the RESULT — Figure 2 is the one that documents the controls —
                # and leaving the panel shut keeps the result panels larger in the
                # same 180 mm.
                try:
                    page.set_input_files("input[type='file']", amostra, timeout=30000)
                    # inference on the free CPU Space is ~350 ms for M, but the upload,
                    # the rerun and the contour plot dominate; wait on the result, not
                    # on a fixed sleep
                    page.wait_for_selector("text=Segmented area", timeout=300000)
                    # the Export buttons and the feedback block render after the
                    # panels; wait on the last of them, not on the first
                    page.wait_for_selector("text=Save feedback", timeout=120000)
                    page.wait_for_timeout(3500)
                    alt = ajusta_altura(page, args.viewport)
                    print(f"    result rendered · content {alt} CSS px")
                except Exception as e:
                    print(f"    [warn] {type(e).__name__}: {e}")

                y0 = 0
                if args.recorte == "resultado":
                    try:
                        cx = page.get_by_text("Result", exact=True).first.bounding_box()
                        if cx:
                            y0 = max(0, int((cx["y"] - 28) * args.scale))
                            print(f"    framing from the Result heading "
                                  f"(dropping {y0} px of Input)")
                    except Exception:
                        print("    [warn] Result heading not located; keeping the "
                              "whole page")
                destino = os.path.join(args.out, "figure7.png")
                grava_com_dpi(page.screenshot(full_page=False), destino, corte_topo=y0)
                tudo_ok &= relata(destino)
                print()

        navegador.close()

    print("=" * 70)
    if tudo_ok:
        print("Both files clear the floor, the line-art resolution and the size limit.")
    else:
        print("Something is short — see the FAIL above. Raising --scale is the fix;\n"
              "resampling the output afterwards is not.")
    print("\nSupply these as SEPARATE FILES. Pasting them into the .docx puts them\n"
          "through Word's 'Print (220 ppi)' compression, which is what reduced every\n"
          "figure currently in the manuscript to 220 ppi.")


if __name__ == "__main__":
    main()
