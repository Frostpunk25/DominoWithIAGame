import os
import random
import time
from domino_gym import DominoEnv
from domino_engine import DominoGame
from sb3_contrib import MaskablePPO

# --- CONFIGURACIÓN ---
NUM_GAMES = 1000            # Número de partidas a simular
MODEL_PATH = "modelos_domino_mask/domino_pro"
NUM_PLAYERS = 2            # 1 vs 1 para un combate limpio (IA vs Bot)
VERBOSE_INTERVAL = 20       # Imprimir progreso cada X partidas

class BenchmarkEnv(DominoEnv):
    """
    Subclase del entorno para forzar la configuración de 2 jugadores
    sin modificar el archivo original domino_gym.py
    """
    def __init__(self, num_players=2):
        super().__init__()
        self.num_players_override = num_players

    def reset(self, seed=None, options=None):
        # Sobrescribimos la instancia del juego antes de resetear
        # para asegurar que sea un duelo 1 vs 1
        self.game = DominoGame(num_players=self.num_players_override, teams=False)
        return super().reset(seed=seed, options=options)

def main():
    print("🚀 INICIANDO BENCHMARK DE IA (Headless)")
    print("=" * 50)
    
    # 1. Verificar Modelo
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"❌ ERROR CRÍTICO: No se encontró el modelo en {MODEL_PATH}.zip")
        print("   Asegúrate de haber entrenado con el script 'train_domino.py' antes.")
        return

    print(f"📂 Cargando modelo: {MODEL_PATH} ...")
    try:
        model = MaskablePPO.load(MODEL_PATH)
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return

    # 2. Crear Entorno de Prueba (1v1)
    print(f"🆚 Configurando entorno: {NUM_PLAYERS} Jugadores (IA vs Random)")
    env = BenchmarkEnv(num_players=NUM_PLAYERS)

    # 3. Contadores
    wins_ai = 0
    wins_random = 0
    
    start_time = time.time()

    print("-" * 50)
    print("⚔️  COMENZANDO SIMULACIÓN DE PARTIDAS...")
    print("-" * 50)

    # 4. Bucle de Entrenamiento / Juego
    for game_idx in range(1, NUM_GAMES + 1):
        obs, _ = env.reset()
        done = False
        
        # Bucle de una partida individual
        while not done:
            current_player = env.game.current_player
            action_masks = env.action_masks()
            
            if current_player == 0:
                # --- TURNO DE LA IA ENTRENADA ---
                action, _ = model.predict(
                    obs, 
                    action_masks=action_masks, 
                    deterministic=True # Importante: jugada consistente
                )
            else:
                # --- TURNO DEL BOT TONTO (Random) ---
                valid_moves = env.game.get_valid_moves(current_player)
                
                if valid_moves:
                    # Elige una jugada aleatoria de las posibles
                    move = random.choice(valid_moves)
                    # Convierte la jugada (ficha, lado) a índice numérico
                    action = env._encode_action(move[0], move[1])
                else:
                    # Si no tiene jugadas, pasa (enviamos acción dummy)
                    # La lógica interna del motor detectará que no es válida y pasará
                    action = 0
            
            # Ejecutar paso
            obs, reward, done, truncated, info = env.step(action)
        
        # --- FIN DE LA PARTIDA ---
        winner = env.game.winner
        
        if winner == 0:
            wins_ai += 1
        else:
            wins_random += 1

        # Reportar progreso
        if game_idx % VERBOSE_INTERVAL == 0:
            current_pct = (wins_ai / game_idx) * 100
            print(f"🎮 Partida {game_idx}/{NUM_GAMES} | WinRate IA: {current_pct:.1f}%")

    # 5. Resultados Finales
    total_time = time.time() - start_time
    print("-" * 50)
    print("📊 RESULTADOS FINALES")
    print("-" * 50)
    
    print(f"🏆 IA Entrenada (Jugador 0): {wins_ai} victorias ({(wins_ai/NUM_GAMES)*100:.2f}%)")
    print(f"🤡 Bot Random   (Jugador 1): {wins_random} victorias ({(wins_random/NUM_GAMES)*100:.2f}%)")
    print(f"⏱️  Tiempo total: {total_time:.2f} segundos")
    print("-" * 50)

    # Conclusión Senior
    ai_win_rate = wins_ai / NUM_GAMES
    if ai_win_rate >= 0.80:
        print("✅ CONCLUSIÓN: La IA es MUY SUPERIOR. ¡Excelente entrenamiento!")
    elif ai_win_rate >= 0.60:
        print("✅ CONCLUSIÓN: La IA es notablemente mejor que el azar.")
    elif ai_win_rate >= 0.51:
        print("⚠️  CONCLUSIÓN: La IA gana, pero por poco margen. Entrena más para mejorar.")
    else:
        print("❌ CONCLUSIÓN: La IA NO supera al azar. Revisa arquitectura o entrena más tiempo.")

if __name__ == "__main__":
    main()