# Convert MP4 videos from mp4v to H.264 for browser compatibility
# Requires ffmpeg: https://ffmpeg.org/download.html

$ffmpegExe = $null
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    $ffmpegExe = "ffmpeg"
} else {
    $wingetPath = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter "ffmpeg.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    if ($wingetPath) { $ffmpegExe = $wingetPath }
}
if (-not $ffmpegExe) {
    Write-Host "ERROR: ffmpeg not found. Install: winget install ffmpeg" -ForegroundColor Red
    exit 1
}

$videos = @(
    "static/videos/image_early_morning.mp4",
    "static/videos/image_foggy.mp4",
    "static/videos/image_overcast.mp4",
    "static/videos/image_rainy.mp4",
    "static/videos/image_snowy.mp4",
    "static/videos/image_sunny.mp4",
    "static/videos/irradiance_early_morning.mp4",
    "static/videos/irradiance_foggy.mp4",
    "static/videos/irradiance_overcast.mp4",
    "static/videos/irradiance_rainy.mp4",
    "static/videos/irradiance_snowy.mp4",
    "static/videos/irradiance_sunny.mp4",
    "static/videos/property_albedo.mp4",
    "static/videos/property_metallic.mp4",
    "static/videos/property_normal.mp4",
    "static/videos/property_roughness.mp4"
)

foreach ($src in $videos) {
    if (Test-Path $src) {
        $tmp = $src + ".h264tmp.mp4"
        Write-Host "Converting: $src"
        & $ffmpegExe -i $src -c:v libx264 -preset fast -crf 23 -movflags +faststart -y $tmp 2>$null
        if ($LASTEXITCODE -eq 0 -and (Test-Path $tmp)) {
            Move-Item $tmp $src -Force
            Write-Host "  OK" -ForegroundColor Green
        } else {
            Write-Host "  Failed" -ForegroundColor Red
            if (Test-Path $tmp) { Remove-Item $tmp }
        }
    }
}

Write-Host "`nDone. Refresh the page to test video playback."
