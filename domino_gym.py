import gymnasium as gym
import numpy as np
from gymnasium import spaces
from domino_engine import DominoGame

class DominoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DominoEnv, self).__init__()
        self.game = DominoGame()
        
        # 110 Acciones posibles
        self.action_space = spaces.Discrete(110)

        # Observación (Mano + Mesa + Historial)
        # Aumentamos un poco el tamaño para futuras estrategias
        self.observation_space = spaces.Box(low=0, high=1, shape=(130,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action_idx):
        # Decodificar acción
        ficha, lado = self._decode_action(action_idx)
        player = self.game.current_player
        
        # Validar jugada (aunque con masking no debería fallar, mantenemos la seguridad)
        valid_moves = self.game.get_valid_moves(player)
        
        # Verificar si la acción elegida es válida
        move_is_valid = False
        chosen_move = (ficha, lado)
        
        # Búsqueda exacta de la jugada
        real_move = None
        for vm in valid_moves:
            f_valid, l_valid = vm
            if set(f_valid) == set(ficha) and l_valid == lado:
                move_is_valid = True
                real_move = vm
                break
        
        if not valid_moves:
            # Si no hay jugadas, forzamos paso (Action masking se encargará de esto también)
            _, game_done = self.game.step(None)
            reward = 0
        elif not move_is_valid:
            # Esto teóricamente NO DEBERÍA PASAR con Masking, pero por seguridad:
            print(f"⚠️ ALERTA: Jugada ilegal saltó el filtro! {chosen_move}")
            return self._get_obs(), -100, True, False, {}
        else:
            # Ejecutar jugada real
            reward_points, game_done = self.game.step(real_move)
            
            # SISTEMA DE RECOMPENSAS MEJORADO
            if game_done:
                if self.game.winner == player:
                    reward = 100 # Ganar es lo único que importa
                else:
                    reward = -10 # Perder duele
            else:
                reward = 0.1 # Pequeño incentivo por seguir jugando
        
        terminated = game_done
        return self._get_obs(), reward, terminated, False, {}

    def action_masks(self):
        """
        NUEVA FUNCIÓN MÁGICA:
        Devuelve una lista de True/False.
        True = Botón habilitado (Jugada válida).
        False = Botón deshabilitado.
        """
        mask = np.zeros(110, dtype=bool)
        player = self.game.current_player
        valid_moves = self.game.get_valid_moves(player)
        
        if not valid_moves:
            # Si no hay jugadas, deberíamos tener una acción de "PASAR".
            # Como nuestro modelo actual fuerza a elegir ficha, 
            # truco: habilitamos la acción 0 pero en step manejamos el paso.
            # O mejor: Si valid_moves está vacío, el motor hace step(None) automáticamente
            # en el bucle de entrenamiento si detecta máscara vacía.
            # Pero para simplificar, dejaremos todo en False y el algoritmo sabrá que debe pasar.
            return mask 

        for move in valid_moves:
            ficha, lado = move
            # Buscar el índice de esta acción
            idx = self._encode_action(ficha, lado)
            if idx < 110:
                mask[idx] = True
                
        return mask

    def _encode_action(self, ficha, lado):
        """Convierte jugada real a índice 0-109"""
        # Ordenamos la ficha para buscarla en la lista maestra
        f_sorted = tuple(sorted(ficha))
        try:
            ficha_idx = self.game.all_pieces.index(f_sorted)
            lado_idx = 0 if lado == 'L' else 1
            return ficha_idx * 2 + lado_idx
        except:
            return 0

    def _get_obs(self):
        player = self.game.current_player
        hand = self.game.hands[player]
        mesa_fichas = self.game.mesa
        extremos = self.game.extremos
        
        hand_vec = np.zeros(55, dtype=np.float32)
        for f in hand:
            hand_vec[self._get_ficha_index(f)] = 1.0
            
        left_vec = np.zeros(10, dtype=np.float32)
        right_vec = np.zeros(10, dtype=np.float32)
        if extremos[0] != -1:
            left_vec[extremos[0]] = 1.0
            right_vec[extremos[1]] = 1.0
            
        board_vec = np.zeros(55, dtype=np.float32)
        for f in mesa_fichas:
            board_vec[self._get_ficha_index(f)] = 1.0
            
        return np.concatenate([hand_vec, left_vec, right_vec, board_vec])

    def _get_ficha_index(self, ficha):
        f_sorted = tuple(sorted(ficha))
        try:
            return self.game.all_pieces.index(f_sorted)
        except:
            return 0

    def _decode_action(self, action_idx):
        ficha_idx = action_idx // 2
        lado_idx = action_idx % 2
        ficha = self.game.all_pieces[ficha_idx]
        lado = 'L' if lado_idx == 0 else 'R'
        return ficha, lado