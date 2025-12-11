import os
from sb3_contrib import MaskablePPO # <--- CAMBIO IMPORTANTE
from sb3_contrib.common.maskable.utils import get_action_masks # Para la demo
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from domino_gym import DominoEnv

# Función auxiliar para envolver el entorno con máscara
def mask_fn(env: DominoEnv):
    return env.action_masks()

def make_env():
    env = DominoEnv()
    return ActionMasker(env, mask_fn)

def main():
    models_dir = "modelos_domino_mask"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print("🚀 Iniciando entrenamiento CON MÁSCARA DE ACCIÓN (Nivel Experto)...")
    
    # Usamos solo 1 entorno primero para asegurar que la máscara funciona, 
    # luego podríamos vectorizar, pero MaskablePPO a veces prefiere entorno simple al inicio.
    # Para tu i9, usaremos 8 entornos vectorizados con Wrapper especial.
    
    # NOTA: Vectorizar ActionMasker es complejo, vamos a hacerlo simple con 1 entorno super rápido
    # o usar dummy vec env. Probemos simple primero para estabilidad.
    env = make_env()

    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device='cpu',
        learning_rate=0.0003,
        gamma=0.99 # Importante para estrategia a largo plazo
    )

    # Entrenamos 100,000 pasos (será mucho más eficiente que 500k sin máscara)
    model.learn(total_timesteps=100_000)
    
    model.save(f"{models_dir}/domino_pro")
    print("✅ Modelo guardado.")

    # --- DEMOSTRACIÓN ---
    print("\n--- DEMO JUGANDO (Sin errores garantizado) ---")
    env_test = DominoEnv() # Entorno puro
    obs, _ = env_test.reset()
    done = False
    
    while not done:
        # Obtener máscaras válidas
        action_masks = env_test.action_masks()
        
        # Si no hay jugadas (máscara vacía), pasamos manual
        if not any(action_masks):
            print("IA tiene que pasar.")
            obs, reward, done, _, _ = env_test.step(0) # Acción dummy, el motor sabe manejar paso
        else:
            # Predecir usando máscara
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            ficha, lado = env_test._decode_action(action)
            print(f"IA juega: {ficha} en {lado}")
            obs, reward, done, _, _ = env_test.step(action)

            if reward == -100:
                print("❌ IMPOSIBLE: Si sale esto, hay un bug en la máscara.")

if __name__ == "__main__":
    main()