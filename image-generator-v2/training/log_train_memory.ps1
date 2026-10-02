# Logs the phase-1 training process's memory every 5 minutes to logs/train_memory.csv,
# alongside the latest training step, to find what makes it grow over time.
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "logs\train_memory.csv"
$steps = Join-Path $root "logs\dit_phase1.csv"
if (-not (Test-Path $out)) { "time,pid,private_gb,working_set_gb,workers_private_gb,available_gb,step" | Out-File $out -Encoding utf8 }
while ($true) {
    $main = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*train.py --phase 1*' } |
        ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue } | Sort-Object WorkingSet64 -Descending | Select-Object -First 1
    if ($main) {
        $workers = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*multiprocessing-fork*' } |
            ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue } | Measure-Object -Property PrivateMemorySize64 -Sum
        $avail = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
        $step = (Get-Content $steps -Tail 1).Split(",")[0]
        "{0},{1},{2:N2},{3:N2},{4:N2},{5:N1},{6}" -f (Get-Date -Format "MM-dd HH:mm"), $main.Id, ($main.PrivateMemorySize64 / 1GB),
            ($main.WorkingSet64 / 1GB), ($workers.Sum / 1GB), $avail, $step | Out-File $out -Append -Encoding utf8
    }
    Start-Sleep -Seconds 300
}
