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

        # Vector de 133: Mano(55) + Extremos(20) + Mesa(55) + Cuentas Rival(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(133,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action_idx):
        ficha, lado = self._decode_action(action_idx)
        player = self.game.current_player
        
        valid_moves = self.game.get_valid_moves(player)
        
        real_move = None
        for vm in valid_moves:
            f_valid, l_valid = vm
            if set(f_valid) == set(ficha) and l_valid == lado:
                real_move = vm
                break
        
        if not valid_moves:
            _, game_done = self.game.step(None)
            reward = 0
        elif not real_move:
            # Seguridad extra
            return self._get_obs(), -10, True, False, {}
        else:
            reward_points, game_done = self.game.step(real_move)
            
            if game_done:
                if self.game.winner == player:
                    reward = 100 
                else:
                    reward = -20 # Penalización fuerte
            else:
                reward = 0.1 
        
        terminated = game_done
        return self._get_obs(), reward, terminated, False, {}

    def action_masks(self):
        mask = np.zeros(110, dtype=bool)
        player = self.game.current_player
        valid_moves = self.game.get_valid_moves(player)
        
        if not valid_moves:
            return mask 

        for move in valid_moves:
            ficha, lado = move
            idx = self._encode_action(ficha, lado)
            if idx < 110:
                mask[idx] = True
        return mask

    def _encode_action(self, ficha, lado):
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
            
        # Observación de oponentes (Normalizada 0-1)
        opp_counts = np.zeros(3, dtype=np.float32)
        if self.game.num_players == 4:
            opp_counts[0] = len(self.game.hands[1]) / 10.0
            opp_counts[1] = len(self.game.hands[2]) / 10.0
            opp_counts[2] = len(self.game.hands[3]) / 10.0
        elif self.game.num_players == 2:
            opp_counts[0] = len(self.game.hands[1]) / 10.0
            opp_counts[1] = 0.0 
            opp_counts[2] = 0.0
            
        return np.concatenate([hand_vec, left_vec, right_vec, board_vec, opp_counts])

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