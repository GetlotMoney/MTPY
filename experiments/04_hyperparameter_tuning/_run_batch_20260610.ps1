# TUNE batch 2026-06-10: baseline topo=0.1 combination tuning
# 10 experiments, TUNE-LITE, no Claude review

 = "experiments/04_hyperparameter_tuning"

 = @(
    @{ id="TUNE-021"; slug="topo01_cond_text001";  desc="topo=0.1 + cond_text=0.01";  config=@{
        lambda_topo_pearson=0.10; conditional_text_ratio=0.010
    }},
    @{ id="TUNE-022"; slug="topo01_jepa_neg_001";  desc="topo=0.1 + jepa_neg=0.01";  config=@{
        lambda_topo_pearson=0.10; lambda_jepa_neg=0.01
    }},
    @{ id="TUNE-023"; slug="topo01_jepa_off";      desc="topo=0.1 + jepa=0.0";       config=@{
        lambda_topo_pearson=0.10; lambda_jepa=0.0
    }},
    @{ id="TUNE-024"; slug="topo01_cond_text001_jepa_neg001"; desc="topo=0.1 + cond_text=0.01 + jepa_neg=0.01"; config=@{
        lambda_topo_pearson=0.10; conditional_text_ratio=0.010; lambda_jepa_neg=0.01
    }},
    @{ id="TUNE-025"; slug="topo01_cond_text001_jepa_off"; desc="topo=0.1 + cond_text=0.01 + jepa=0.0"; config=@{
        lambda_topo_pearson=0.10; conditional_text_ratio=0.010; lambda_jepa=0.0
    }},
    @{ id="TUNE-026"; slug="topo01_jepa_off_jepa_neg001"; desc="topo=0.1 + jepa=0.0 + jepa_neg=0.01"; config=@{
        lambda_topo_pearson=0.10; lambda_jepa=0.0; lambda_jepa_neg=0.01
    }},
    @{ id="TUNE-027"; slug="topo015";               desc="topo=0.15";                 config=@{
        lambda_topo_pearson=0.15
    }},
    @{ id="TUNE-028"; slug="topo008";               desc="topo=0.08";                 config=@{
        lambda_topo_pearson=0.08
    }},
    @{ id="TUNE-029"; slug="topo01_lastvit_k16";    desc="topo=0.1 + lastvit_k=16";   config=@{
        lambda_topo_pearson=0.10; lastvit_select_k=16
    }},
    @{ id="TUNE-030"; slug="topo01_cond_text001_lastvit_k16"; desc="topo=0.1 + cond_text=0.01 + lastvit_k=16"; config=@{
        lambda_topo_pearson=0.10; conditional_text_ratio=0.010; lastvit_select_k=16
    }}
)

Write-Host "=== 10 TUNE experiments ready ==="
 | ForEach-Object { Write-Host "_: " }
