# .latexmkrc configuration
$pdf_mode = 1;
$postscript_mode = 0;
$dvi_mode = 0;

# Trigger Okular reload safely via D-Bus only after compilation finishes successfully
$success_cmd = 'which qdbus >/dev/null 2>&1 && qdbus 2>/dev/null | grep -o "org.kde.okular-[0-9]*" | while read -r id; do qdbus "$id" /okular org.kde.okular.reload 2>/dev/null; done || true';
