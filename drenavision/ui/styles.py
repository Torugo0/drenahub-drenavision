"""Small visual system for the citizen-facing inspection workspace."""

PAGE_STYLE = """
<style>
:root { --dv-navy:#102e52; --dv-blue:#1468d4; --dv-cyan:#2eb7de; --dv-muted:#61758c; }
.stApp { background:#f1f6fc; }
.block-container { max-width:1440px; padding:1.25rem 2rem 2rem; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"] { display:none; }
[data-testid="stMainBlockContainer"] { padding-top:1.25rem; }
.dv-hero { display:flex; align-items:center; gap:16px; padding:12px 24px; margin:0 0 12px; border-radius:20px; color:white; background:linear-gradient(112deg,#102e52 0%,#125aa3 70%,#168cbd 100%); box-shadow:0 9px 25px #123f6b14; }
.dv-logo { width:53px;height:53px;display:grid;place-items:center;border-radius:17px;background:#ffffff20;font-size:29px;flex-shrink:0; }
.dv-hero h1 { padding:0;margin:2px 0 4px;color:#fff;font-size:28px;line-height:1.15;letter-spacing:-.7px; }
.dv-hero p { padding:0;margin:0;color:#d9ecff;font-size:14px; }
.dv-eyebrow { color:#a9dcff;font-size:10px;font-weight:700;letter-spacing:1.5px; }
.dv-hero-note { margin-left:auto;color:#d4ebff;font-size:12px;max-width:190px;line-height:1.5; }
[data-testid="stTabs"] [role="tablist"] { gap:24px;border-bottom:1px solid #d8e4f2;padding-left:4px;margin-bottom:10px; }
[data-testid="stTabs"] [role="tab"] { font-weight:600;padding:6px 0 12px; }
[data-testid="stTabs"] [aria-selected="true"] { color:#1468d4; }
h2,h3 { color:#102e52;letter-spacing:-.35px; }
h3 { font-size:1.15rem!important;padding-top:0!important; }
p, label { color:#264664; }
[data-testid="stCaptionContainer"] { color:#61758c;font-size:12px; }
.st-key-inspection_input, .st-key-result_panel { background:#fff;border:1px solid #dbe6f3;border-radius:18px;padding:20px;box-shadow:0 4px 16px #123f6b05; }
.st-key-inspection_input [data-testid="stVerticalBlock"], .st-key-result_panel [data-testid="stVerticalBlock"] { gap:.65rem; }
.st-key-inspection_input [data-testid="stFileUploaderDropzone"] { border:1px dashed #9abde8;background:#f5f9ff;border-radius:12px; }
.st-key-inspection_input [data-testid="stFileUploaderDropzoneInstructions"] { font-size:12px; }
[data-testid="stTextInput"] input, [data-testid="stSelectbox"] [role="combobox"] { background:#f7faff;border-color:#dae7f5;border-radius:10px; }
[data-testid="stBaseButton-primary"] { border:0;border-radius:10px;background:linear-gradient(90deg,#1266cc,#168bc1);min-height:42px;font-weight:600; }
[data-testid="stBaseButton-primary"] p { color:white!important; }
[data-testid="stBaseButton-primary"]:disabled { opacity:.5; }
[data-testid="stBaseButton-secondary"] { border-color:#cedff2;border-radius:9px;color:#155fae;background:#fff; }
[data-testid="stAlert"] { padding:11px 13px;border-radius:12px; }
[data-testid="stAlert"] p { font-size:13px; }
.dv-counts { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:1px 0 8px; }
.dv-count { border:1px solid #e0eaf6;border-radius:11px;background:#f7faff;padding:9px 11px; }
.dv-count span { display:block;font-size:11px;color:#61758c; }
.dv-count strong { font-size:18px;color:#163c67;line-height:1.4; }
.dv-count.active { border-color:#b5d5fa;background:#eaf4ff; }
.dv-count.active strong { color:#1266cc; }
.dv-count small { font-size:10px;color:#61758c;margin-left:4px; }
.st-key-result_panel [data-testid="stImage"] { display:flex;justify-content:center;background:#edf3fa;border-radius:13px;overflow:hidden; }
.st-key-result_panel [data-testid="stImage"] img { width:100%!important;height:min(38vh,360px)!important;min-height:220px;object-fit:contain;background:#edf3fa; }
.st-key-result_panel video { display:block;width:100%!important;height:min(38vh,360px)!important;min-height:220px;object-fit:contain;background:#0d2036;border-radius:13px; }
.st-key-result_panel [data-testid="stVideo"] { width:100%; }
.dv-empty { min-height:420px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:25px;color:#61758c; }
.dv-empty-icon { width:78px;height:78px;border-radius:24px;background:#eaf4ff;display:grid;place-items:center;color:#2289dc;font-size:37px;margin-bottom:18px; }
.dv-empty h3 { margin:0 0 8px;font-size:19px!important; }
.dv-empty p { margin:0;max-width:310px;line-height:1.6;color:#61758c;font-size:14px; }
.dv-footnote { margin-top:14px;padding:10px 3px;font-size:11px;color:#71869c; }
[data-testid="stExpander"] { border-color:#e0e9f4;border-radius:10px; }
[data-testid="stExpander"] summary p { font-size:12px; }
.dv-history { border:1px solid #dce7f4;background:#fff;border-radius:13px;padding:13px 16px;margin-bottom:10px; }
.dv-history strong { color:#143d66; }
.dv-history p { margin:4px 0;font-size:13px; }
.dv-history small { color:#6f8299;font-size:11px; }
@media(max-width:900px) {
 .block-container { padding:1rem 1rem 2rem; }
 .dv-hero { border-radius:16px;padding:17px;gap:12px; }
 .dv-hero-note { display:none; }
 .dv-hero h1 { font-size:24px; }
 .st-key-analysis_workspace [data-testid="stHorizontalBlock"] { flex-direction:column; }
 .st-key-analysis_workspace [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width:100%!important;flex:1 1 100%!important; }
 .st-key-inspection_input,.st-key-result_panel { padding:17px; }
 .dv-empty { min-height:210px; }
 .st-key-result_panel video,.st-key-result_panel [data-testid="stImage"] img { height:320px!important;min-height:0; }
}
@media(max-width:480px) {
 .dv-hero p { font-size:12px; }
 .dv-count { padding:8px; }
 .dv-count small { display:none; }
 [data-testid="stTabs"] [role="tablist"] { gap:15px; }
 [data-testid="stTabs"] [role="tab"] p { font-size:12px; }
}
</style>
"""

HERO = """
<div class="dv-hero">
  <div class="dv-logo">💧</div>
  <div><div class="dv-eyebrow">DRENAHUB · INSPEÇÃO VISUAL</div>
  <h1>DrenaVision</h1><p>Cuidado com a drenagem, em um só olhar.</p></div>
  <div class="dv-hero-note">Foto ou vídeo → análise visual → verificação no local</div>
</div>
"""

EMPTY_RESULT = """
<div class="dv-empty"><div class="dv-empty-icon">⌕</div>
<h3>Seu resultado aparece aqui</h3>
<p>Escolha um registro ao lado e toque em <strong>Analisar foto ou vídeo</strong>.
Vamos destacar os sinais encontrados.</p></div>
"""
