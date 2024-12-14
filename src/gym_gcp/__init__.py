from gymnasium.envs.registration import register

register(
    id="GCP-v0",
    entry_point="gym_gcp.envs:GCP_Env"
)
