import os
import time
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv # Importante para Multiproceso
from stable_baselines3.common.callbacks import CheckpointCallback
from domino_gym import DominoEnv

# --- OPTIMIZACIÓN DE CPU PARA i9-13900H ---
# Al usar multiproceso (SubprocVecEnv), NO queremos que PyTorch use 
# múltiples hilos internamente o saturaremos la CPU (Threading Oversubscription).
# Forzamos 1 hilo por proceso. La paralelización real vendrá de los procesos.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
# ------------------------------------------

# Directorios
models_dir = "modelos_domino_mask"
logs_dir = "logs_domino_mask"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

def mask_fn(env: DominoEnv):
    return env.action_masks()

def make_env(rank):
    """
    Función factory para crear entornos con semilla única por proceso.
    rank: índice del proceso (0 a 11)
    """
    def _init():
        # Importamos dentro para evitar problemas con 'spawn' en Windows/Linux
        from domino_gym import DominoEnv
        env = DominoEnv()
        env = ActionMasker(env, mask_fn)
        return env
    return _init

def main():
    print(f"🚀 Iniciando entrenamiento OPTIMIZADO PARA i9-13900H")
    print(f"🔧 Modo: Multiproceso Real (Subprocess)")
    
    # NÚMERO DE TRABAJADORES (NUM_ENVS)
    # Un i9-13900H tiene 14 núcleos. Usamos 12 para dejar margen al sistema.
    # Esto creará 12 partidas jugando simultáneamente.
    num_envs = 12 
    
    print(f"⚡ {num_envs} Entornos paralelos activos")
    
    # Usamos SubprocVecEnv para eludir el GIL de Python
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Configuración del Modelo
    # learning_rate lento para estabilidad
    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device='cpu', # Mantenemos CPU por compatibilidad, el i9 es una bestia en FP32
        learning_rate=0.0001,
        gamma=0.99,
        tensorboard_log=logs_dir
    )

    # Callback para guardar checkpoints cada 200k pasos (ahora que es más rápido)
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000, 
        save_path=logs_dir,
        name_prefix="domino_checkpoint"
    )

    start_time = time.time()
    
    print("📊 Comenzando entrenamiento de 1,000,000 pasos...")
    
    # ENTRENAMIENTO
    model.learn(
        total_timesteps=1_000_000, 
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    total_time = time.time() - start_time
    print(f"✅ Entrenamiento completado en {total_time/60:.2f} minutos.")
    
    # Guardar modelo final
    model.save(f"{models_dir}/domino_pro")
    print("✅ Modelo final guardado.")

    # --- TEST RÁPIDO ---
    print("\n--- TEST DE VERIFICACIÓN ---")
    # Usamos un solo entorno para probar al final
    env_test = make_env(0)()
    obs, _ = env_test.reset()
    
    for i in range(5): 
        action_masks = env_test.action_masks()
        if not any(action_masks):
            print("Paso")
            obs, _, done, _, _ = env_test.step(0)
        else:
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, _, done, _, _ = env_test.step(action)
        if done: break

if __name__ == "__main__":
    # Requerido para SubprocVecEnv en Windows
    main()