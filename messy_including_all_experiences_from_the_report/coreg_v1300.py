from caveclient import CAVEclient

client = CAVEclient("minnie65_public")
client.version = 1300

coreg = client.materialize.query_table(
    "coregistration_auto_phase3_fwd_apl_vess_combined_v2",
    materialization_version=1300
)
coreg.to_csv("coreg_v1300.csv", index=False)