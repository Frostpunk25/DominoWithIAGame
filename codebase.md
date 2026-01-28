# domino_engine.py

```py
import random

class DominoGame:
    def __init__(self, num_players=4, teams=False):
        self.num_players = num_players
        self.teams = teams
        self.all_pieces = [(i, j) for i in range(10) for j in range(i, 10)]
        self.reset()

    def reset(self):
        # 1. Barajar REALMENTE bien
        self.piezas = self.all_pieces[:]
        random.shuffle(self.piezas)
        
        # 2. Repartir
        self.hands = {}
        for p in range(self.num_players):
            start = p * 10
            end = start + 10
            self.hands[p] = self.piezas[start:end]
            
        self.mesa = [] 
        self.extremos = [-1, -1] 
        
        # Historial separado para dibujar ramas izquierda/derecha
        self.history_left = []  
        self.history_right = [] 
        self.center_tile = None 
        
        # 3. Decidir quién sale
        self.current_player, self.start_reason = self._find_starting_player()
        
        self.winner = -1
        self.game_over = False
        self.pass_count = 0
        
        return self._get_state()

    def _find_starting_player(self):
        # Prioridad: Doble más alto
        for d in range(9, -1, -1):
            ficha = (d, d)
            for p in range(self.num_players):
                if ficha in self.hands[p]:
                    return p, f"Salida por Doble {d}"
        
        # Si nadie tiene dobles (raro en doble 9), aleatorio
        starter = random.randint(0, self.num_players - 1)
        return starter, "Sorteo Aleatorio (Nadie tenía dobles)"

    def get_valid_moves(self, player):
        hand = self.hands[player]
        if self.center_tile is None:
            return [(f, 'L') for f in hand] 

        valid = []
        l_val, r_val = self.extremos
        
        for f in hand:
            v1, v2 = f
            # Detectar coincidencias
            matches_l = (v1 == l_val or v2 == l_val)
            matches_r = (v1 == r_val or v2 == r_val)
            
            if matches_l: valid.append((f, 'L'))
            
            # Solo agregamos R si es distinto a L o si es la misma ficha pero conecta al otro lado
            if matches_r:
                # Evitar duplicar la misma jugada exacta si l_val == r_val
                if l_val != r_val or not matches_l:
                    valid.append((f, 'R'))
                elif l_val == r_val and matches_l:
                    # Caso especial: 6-6 en mesa, tengo 6-1. Pega por los dos lados igual.
                    # Agregamos R explícitamente para permitir la elección del usuario
                    valid.append((f, 'R'))
                    
        return valid

    def step(self, action):
        player = self.current_player
        
        if action is None:
            self.pass_count += 1
        else:
            ficha, lado = action
            if ficha not in self.hands[player]:
                return -100, True 

            self.hands[player].remove(ficha)
            
            # --- FIX: Actualizar la lista 'mesa' ---
            self.mesa.append(ficha)
            # -------------------------------------

            if self.center_tile is None:
                self.center_tile = ficha
                self.extremos = [ficha[0], ficha[1]]
            else:
                target = self.extremos[0] if lado == 'L' else self.extremos[1]
                v1, v2 = ficha
                
                if v1 == target:
                    nuevo = v2
                    conector = v1
                else:
                    nuevo = v1
                    conector = v2
                
                move_data = {
                    'ficha': ficha, 
                    'player': player, 
                    'conector': conector, 
                    'nuevo_extremo': nuevo
                }
                
                if lado == 'L':
                    self.extremos[0] = nuevo
                    self.history_left.append(move_data)
                else:
                    self.extremos[1] = nuevo
                    self.history_right.append(move_data)
            
            self.pass_count = 0

        if len(self.hands[player]) == 0:
            self.winner = player
            self.game_over = True
            return 100, True
        
        if self.pass_count >= self.num_players:
            self.game_over = True
            self.winner = self._calculate_winner_by_points()
            return 0, True
            
        self.current_player = (self.current_player + 1) % self.num_players
        return 0, False

    def _calculate_winner_by_points(self):
        sums = {p: sum(f[0]+f[1] for f in h) for p, h in self.hands.items()}
        if self.teams and self.num_players == 4:
            t1 = sums[0] + sums[2]
            t2 = sums[1] + sums[3]
            return 0 if t1 < t2 else 1
        return min(sums, key=sums.get)
        
    def _get_state(self): return {}
```

# domino_gym.py

```py
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
```

# DominoWithIAGame.zip

This is a binary file of the type: Compressed Archive

# gui_domino.py

```py
import pygame
import sys
import random
from sb3_contrib import MaskablePPO
from domino_gym import DominoEnv
from domino_engine import DominoGame

# --- CONFIGURACIÓN VISUAL ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BG_COLOR = (30, 90, 30) # Verde clásico
TILE_COLOR = (245, 245, 235)
DOT_COLOR = (15, 15, 15)
HIGHLIGHT = (255, 215, 0)

# Dimensiones de fichas
TILE_W = 38
TILE_H = 76
GAP = 2

# Márgenes de seguridad para que la "Serpiente" no choque con las manos
MARGIN_TOP = 120
MARGIN_BOTTOM = 150
MARGIN_LEFT = 100
MARGIN_RIGHT = 100

class DominoGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Dominó Pro v3 - Snake AI (Espiral Fixed)")
        self.font = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.big_font = pygame.font.SysFont("Segoe UI", 40, bold=True)
        self.clock = pygame.time.Clock()
        
        print("Cargando IA...")
        try:
            self.model = MaskablePPO.load("modelos_domino_mask/domino_pro")
        except:
            self.model = None
            print("⚠️ Modo Random (Modelo no encontrado)")

        self.state = "MENU"
        self.game = None
        self.tile_rects = [] # Para guardar colisiones de la mano
        
    def draw_pips(self, surface, x, y, number, size, vertical):
        """Dibuja los puntos con precisión matemática"""
        if vertical:
            cx, cy = x + size//2, y + size//2
        else:
            cx, cy = x + size//2, y + size//2
            
        offset = size // 4
        r = 3
        
        pips_map = [
            (0,0), (-1,-1), (1,1), (1,-1), (-1,1), 
            (-1,0), (1,0), (0,-1), (0,1)
        ]
        
        active = []
        if number==1: active=[0]
        elif number==2: active=[1,2]
        elif number==3: active=[0,1,2]
        elif number==4: active=[1,2,3,4]
        elif number==5: active=[0,1,2,3,4]
        elif number==6: active=[1,2,3,4,5,6]
        elif number==7: active=[0,1,2,3,4,5,6]
        elif number==8: active=[1,2,3,4,5,6,7,8]
        elif number==9: active=[0,1,2,3,4,5,6,7,8]

        for i in active:
            px, py = pips_map[i]
            pygame.draw.circle(surface, DOT_COLOR, (int(cx + px*offset), int(cy + py*offset)), r)

    def draw_tile_graphic(self, x, y, v1, v2, vertical=True, selected=False):
        w, h = (TILE_W, TILE_H) if vertical else (TILE_H, TILE_W)
        rect = pygame.Rect(x, y, w, h)
        
        pygame.draw.rect(self.screen, (20,20,20), (x+2, y+2, w, h), border_radius=4)
        pygame.draw.rect(self.screen, TILE_COLOR, rect, border_radius=4)
        
        color_borde = HIGHLIGHT if selected else (80,80,80)
        pygame.draw.rect(self.screen, color_borde, rect, 2 if selected else 1, border_radius=4)
        
        half = TILE_W
        
        if vertical:
            pygame.draw.line(self.screen, (150,150,150), (x+4, y+h//2), (x+w-4, y+h//2), 1)
            self.draw_pips(self.screen, x, y, v1, half, True)
            self.draw_pips(self.screen, x, y+h//2, v2, half, True)
        else:
            pygame.draw.line(self.screen, (150,150,150), (x+w//2, y+4), (x+w//2, y+h-4), 1)
            self.draw_pips(self.screen, x, y, v1, half, False)
            self.draw_pips(self.screen, x+w//2, y, v2, half, False)
        
        return rect

    def calculate_snake_layout(self, history, start_x, start_y, start_direction):
        """
        Lógica de Espiral:
        - Rama Derecha (dir=1): Empieza derecha -> Gira ARRIBA -> Gira ABAJO...
        - Rama Izquierda (dir=-1): Empieza izquierda -> Gira ABAJO -> Gira ARRIBA...
        """
        layout = []
        curr_x, curr_y = start_x, start_y
        direction = start_direction # 1=Der, -1=Izq
        
        # Configurar dirección vertical inicial según la rama
        # Si empieza a la derecha, el primer giro es hacia ARRIBA (-1)
        # Si empieza a la izquierda, el primer giro es hacia ABAJO (1)
        vertical_dir = -1 if start_direction == 1 else 1
        
        step_long = TILE_H + GAP
        step_short = TILE_W + GAP
        
        for move in history:
            ficha = move['ficha']
            conector = move['conector']
            nuevo = move['nuevo_extremo']
            is_double = (ficha[0] == ficha[1])
            
            # 1. Determinar posición de dibujo actual
            draw_x, draw_y = curr_x, curr_y
            draw_vertical = False
            val_left, val_right = 0, 0
            offset = 0
            
            if is_double:
                draw_vertical = True
                # Ajuste vertical para centrar dobles
                draw_y = curr_y - (TILE_H - TILE_W)//2
                
                if direction == -1:
                    draw_x = curr_x - TILE_W 
                
                val_left, val_right = ficha[0], ficha[1]
                offset = step_short
                
            else:
                draw_vertical = False
                if direction == -1:
                    draw_x = curr_x - TILE_H
                
                if direction == 1:
                    val_left, val_right = conector, nuevo
                else:
                    val_left, val_right = nuevo, conector
                
                offset = step_long

            layout.append({
                'x': draw_x, 'y': draw_y, 
                'v1': val_left, 'v2': val_right, 
                'vert': draw_vertical
            })
            
            # 2. Avanzar cursor
            curr_x += (offset * direction)
            
            # 3. Verificar colisión y ajustar para el SIGUIENTE paso (Lógica de Espiral)
            # Verificamos si el PROXIMO cursor se saldría
            limit_right = SCREEN_WIDTH - MARGIN_RIGHT
            limit_left = MARGIN_LEFT
            
            # Predicción de posición futura para detectar choque
            next_x = curr_x + (step_long * direction)
            
            hit_right = (direction == 1) and (next_x > limit_right)
            hit_left = (direction == -1) and (next_x < limit_left)
            
            if hit_right or hit_left:
                # Girar horizontalmente
                direction *= -1
                
                # Alinear X al borde opuesto para empate perfecto
                if direction == 1: # Ahora vamos a la derecha (estábamos en el borde izq)
                    curr_x = limit_left
                else: # Ahora vamos a la izquierda (estábamos en el borde der)
                    curr_x = limit_right
                
                # Aplicar movimiento vertical (Alternar según espiral)
                curr_y += (step_long * vertical_dir)
                vertical_dir *= -1 # Invertir para el próximo giro
            
        return layout

    def draw_board(self):
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        
        if self.game.center_tile is None: return

        # 1. Centro
        c_ficha = self.game.center_tile
        c_vert = (c_ficha[0] == c_ficha[1])
        
        start_x = cx - (TILE_W//2 if c_vert else TILE_H//2)
        start_y = cy - (TILE_W//2) 
        
        self.draw_tile_graphic(start_x, start_y, c_ficha[0], c_ficha[1], c_vert)
        
        # 2. Calcular Ramas
        # Rama Derecha
        off_x = (TILE_W if c_vert else TILE_H) + GAP
        r_x = start_x + off_x
        r_y = start_y + (TILE_H//2 if c_vert else TILE_W//2) - TILE_W//2
        
        layout_r = self.calculate_snake_layout(self.game.history_right, r_x, r_y, 1)
        for item in layout_r:
            self.draw_tile_graphic(item['x'], item['y'], item['v1'], item['v2'], item['vert'])
            
        # Rama Izquierda
        l_x = start_x - GAP
        l_y = start_y + (TILE_H//2 if c_vert else TILE_W//2) - TILE_W//2
        
        layout_l = self.calculate_snake_layout(self.game.history_left, l_x, l_y, -1)
        for item in layout_l:
            self.draw_tile_graphic(item['x'], item['y'], item['v1'], item['v2'], item['vert'])

    def draw_hands(self):
        self.tile_rects = [] 
        
        # Humano (Abajo)
        hand = self.game.hands[0]
        total_w = len(hand) * (TILE_W + 5)
        start_x = (SCREEN_WIDTH - total_w) // 2
        y = SCREEN_HEIGHT - TILE_H - 20
        
        for i, f in enumerate(hand):
            sel = (i == self.selected_tile_idx)
            offset = -15 if sel else 0
            rect = self.draw_tile_graphic(start_x + i*(TILE_W+5), y + offset, f[0], f[1], True, sel)
            self.tile_rects.append((rect, i, f))
            
        # Bots (Solo dorsos)
        for p in range(1, self.game.num_players):
            n = len(self.game.hands[p])
            if p==1: # Der
                h = n*25
                sy = (SCREEN_HEIGHT - h)//2
                sx = SCREEN_WIDTH - 50
                for k in range(n): pygame.draw.rect(self.screen, (80,60,40), (sx, sy+k*25, 40, 20), border_radius=3)
            elif p==2: # Arr
                w = n*25
                sx = (SCREEN_WIDTH - w)//2
                sy = 20
                for k in range(n): pygame.draw.rect(self.screen, (80,60,40), (sx+k*25, sy, 20, 40), border_radius=3)
            elif p==3: # Izq
                h = n*25
                sy = (SCREEN_HEIGHT - h)//2
                sx = 10
                for k in range(n): pygame.draw.rect(self.screen, (80,60,40), (sx, sy+k*25, 40, 20), border_radius=3)

    def draw_menu(self):
        self.screen.fill((20, 20, 25))
        t = self.big_font.render("DOMINÓ PRO IA", True, HIGHLIGHT)
        self.screen.blit(t, (SCREEN_WIDTH//2 - t.get_width()//2, 100))
        
        opts = [("1 vs 1", 2, False), ("4 Jugadores (FFA)", 4, False), ("2 vs 2 (Equipos)", 4, True)]
        mx, my = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]
        
        for i, (txt, n, tm) in enumerate(opts):
            rect = pygame.Rect(SCREEN_WIDTH//2 - 150, 250 + i*90, 300, 70)
            col = (50, 100, 50) if rect.collidepoint(mx, my) else (40, 40, 50)
            pygame.draw.rect(self.screen, col, rect, border_radius=10)
            pygame.draw.rect(self.screen, (200,200,200), rect, 2, border_radius=10)
            
            surf = self.font.render(txt, True, (255,255,255))
            self.screen.blit(surf, (rect.centerx - surf.get_width()//2, rect.centery - surf.get_height()//2))
            
            if click and rect.collidepoint(mx, my):
                self.game = DominoGame(n, tm)
                self.state = "PLAY"
                self.selected_tile_idx = None
                pygame.time.delay(200)

    def run(self):
        while True:
            if self.state == "MENU":
                self.draw_menu()
                for e in pygame.event.get():
                    if e.type == pygame.QUIT: pygame.quit(); sys.exit()
                pygame.display.flip()
                
            elif self.state == "PLAY":
                self.screen.fill(BG_COLOR)
                self.draw_board()
                self.draw_hands()
                
                # GAME OVER MODAL
                if self.game.game_over:
                    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                    overlay.fill((0,0,0, 180))
                    self.screen.blit(overlay, (0,0))
                    
                    win_txt = f"¡VICTORIA!" if self.game.winner == 0 else f"GANADOR: JUGADOR {self.game.winner}"
                    col = (0, 255, 0) if self.game.winner == 0 else (255, 100, 100)
                    surf = self.big_font.render(win_txt, True, col)
                    self.screen.blit(surf, (SCREEN_WIDTH//2 - surf.get_width()//2, SCREEN_HEIGHT//2 - 50))
                    
                    sub = self.font.render("Click para volver al Menú", True, (255,255,255))
                    self.screen.blit(sub, (SCREEN_WIDTH//2 - sub.get_width()//2, SCREEN_HEIGHT//2 + 20))
                  
                # Info HUD
                if not self.game.game_over:
                    turn_txt = f"Turno: {'TÚ' if self.game.current_player==0 else f'BOT {self.game.current_player}'}"
                    self.screen.blit(self.font.render(turn_txt, True, (255,255,255)), (20, SCREEN_HEIGHT-100))
                    
                    if len(self.game.mesa) == 0:
                        start_info = getattr(self.game, 'start_reason', '')
                        st = self.font.render(start_info, True, HIGHLIGHT)
                        self.screen.blit(st, (20, 20))
                
                # Lógica del Juego
                if not self.game.game_over:
                    turn = self.game.current_player
                    if turn == 0: # Humano
                        if not self.game.get_valid_moves(0):
                            # Paso automático visual
                            self.screen.blit(self.big_font.render("¡PASO!", True, (255,0,0)), (SCREEN_WIDTH//2-50, SCREEN_HEIGHT-200))
                            pygame.display.flip()
                            pygame.time.delay(500)
                            self.game.step(None)
                    else: # IA
                        pygame.display.flip()
                        pygame.time.delay(500)
                        moves = self.game.get_valid_moves(turn)
                        if moves:
                            move = random.choice(moves)
                            self.game.step(move)
                        else:
                            self.game.step(None)

                # Eventos
                for e in pygame.event.get():
                    if e.type == pygame.QUIT: pygame.quit(); sys.exit()
                    
                    if e.type == pygame.MOUSEBUTTONDOWN:
                        if self.game.game_over:
                            self.state = "MENU" 
                            self.game = None
                        
                        elif self.game.current_player == 0:
                            mx, my = pygame.mouse.get_pos()
                            
                            for rect, idx, ficha in self.tile_rects:
                                if rect.collidepoint(mx, my):
                                    self.selected_tile_idx = idx
                                    
                                    valid = self.game.get_valid_moves(0)
                                    possible_moves = [m for m in valid if m[0] == ficha]
                                    
                                    if not possible_moves:
                                        pass 
                                    elif len(possible_moves) == 1:
                                        self.game.step(possible_moves[0])
                                        self.selected_tile_idx = None
                                    else:
                                        rel_x = mx - rect.x
                                        is_left_click = rel_x < (rect.width / 2)
                                        
                                        move_to_play = None
                                        has_L = any(m[1] == 'L' for m in possible_moves)
                                        has_R = any(m[1] == 'R' for m in possible_moves)
                                        
                                        if is_left_click and has_L:
                                            move_to_play = (ficha, 'L')
                                        elif not is_left_click and has_R:
                                            move_to_play = (ficha, 'R')
                                        else:
                                            move_to_play = possible_moves[0]
                                            
                                        self.game.step(move_to_play)
                                        self.selected_tile_idx = None

                pygame.display.flip()
                self.clock.tick(30)

if __name__ == "__main__":
    DominoGUI().run()
```

# modelos_domino_mask\domino_pro.zip

This is a binary file of the type: Compressed Archive

# README.txt

```txt
Alejandro Javier Morejón Santiesteban

```

# train_domino.py

```py
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
```

