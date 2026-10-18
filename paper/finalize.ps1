param([string]$Raw, [string]$OutDocx, [string]$OutPdf)
$marker = [string][char]0x00A7 + "EQ" + [string][char]0x00A7
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
    $doc = $w.Documents.Open($Raw, $false, $false)
    $n = 0
    $count = $doc.Paragraphs.Count
    for ($i = 1; $i -le $count; $i++) {
        $p = $doc.Paragraphs.Item($i)
        $t = $p.Range.Text
        if ($t.StartsWith($marker)) {
            $r = $p.Range
            [void]$r.MoveEnd(1, -1)
            $r.Text = $t.Substring($marker.Length).TrimEnd("`r", "`n")
            $m = $r.OMaths.Add($r)
            $m.OMaths.Item(1).BuildUp()
            $m.Font.Size = 10
            $n++
        }
    }
    Write-Output "equations converted: $n"
    $doc.SaveAs2($OutDocx, 16)
    $doc.SaveAs2($OutPdf, 17)
    Write-Output ("pages: " + $doc.ComputeStatistics(2))
    $doc.Close($false)
} finally {
    $w.Quit()
}
