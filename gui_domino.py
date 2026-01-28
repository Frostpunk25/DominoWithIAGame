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

# Márgenes de seguridad
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
        self.tile_rects = [] 
        self.selected_tile_idx = None
        
        # FIX: Variable para evitar el "Ghost Click" al volver al menú
        self.last_click_time = 0 

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
        layout = []
        curr_x, curr_y = start_x, start_y
        direction = start_direction 
        vertical_dir = -1 if start_direction == 1 else 1
        step_long = TILE_H + GAP
        step_short = TILE_W + GAP
        
        for move in history:
            ficha = move['ficha']
            conector = move['conector']
            nuevo = move['nuevo_extremo']
            is_double = (ficha[0] == ficha[1])
            
            draw_x, draw_y = curr_x, curr_y
            draw_vertical = False
            val_left, val_right = 0, 0
            offset = 0
            
            if is_double:
                draw_vertical = True
                draw_y = curr_y - (TILE_H - TILE_W)//2
                if direction == -1: draw_x = curr_x - TILE_W 
                val_left, val_right = ficha[0], ficha[1]
                offset = step_short
            else:
                draw_vertical = False
                if direction == -1: draw_x = curr_x - TILE_H
                if direction == 1: val_left, val_right = conector, nuevo
                else: val_left, val_right = nuevo, conector
                offset = step_long

            layout.append({'x': draw_x, 'y': draw_y, 'v1': val_left, 'v2': val_right, 'vert': draw_vertical})
            curr_x += (offset * direction)
            
            limit_right = SCREEN_WIDTH - MARGIN_RIGHT
            limit_left = MARGIN_LEFT
            next_x = curr_x + (step_long * direction)
            hit_right = (direction == 1) and (next_x > limit_right)
            hit_left = (direction == -1) and (next_x < limit_left)
            
            if hit_right or hit_left:
                direction *= -1
                if direction == 1: curr_x = limit_left
                else: curr_x = limit_right
                curr_y += (step_long * vertical_dir)
                vertical_dir *= -1
        return layout

    def draw_board(self):
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        if self.game.center_tile is None: return

        c_ficha = self.game.center_tile
        c_vert = (c_ficha[0] == c_ficha[1])
        start_x = cx - (TILE_W//2 if c_vert else TILE_H//2)
        start_y = cy - (TILE_W//2) 
        self.draw_tile_graphic(start_x, start_y, c_ficha[0], c_ficha[1], c_vert)
        
        off_x = (TILE_W if c_vert else TILE_H) + GAP
        r_x = start_x + off_x
        r_y = start_y + (TILE_H//2 if c_vert else TILE_W//2) - TILE_W//2
        layout_r = self.calculate_snake_layout(self.game.history_right, r_x, r_y, 1)
        for item in layout_r: self.draw_tile_graphic(item['x'], item['y'], item['v1'], item['v2'], item['vert'])
            
        l_x = start_x - GAP
        l_y = start_y + (TILE_H//2 if c_vert else TILE_W//2) - TILE_W//2
        layout_l = self.calculate_snake_layout(self.game.history_left, l_x, l_y, -1)
        for item in layout_l: self.draw_tile_graphic(item['x'], item['y'], item['v1'], item['v2'], item['vert'])

    def draw_hands(self):
        self.tile_rects = [] 
        hand = self.game.hands[0]
        total_w = len(hand) * (TILE_W + 5)
        start_x = (SCREEN_WIDTH - total_w) // 2
        y = SCREEN_HEIGHT - TILE_H - 20
        
        for i, f in enumerate(hand):
            sel = (i == self.selected_tile_idx)
            offset = -15 if sel else 0
            rect = self.draw_tile_graphic(start_x + i*(TILE_W+5), y + offset, f[0], f[1], True, sel)
            self.tile_rects.append((rect, i, f))
            
        for p in range(1, self.game.num_players):
            n = len(self.game.hands[p])
            if p==1:
                h = n*25; sy = (SCREEN_HEIGHT - h)//2; sx = SCREEN_WIDTH - 50
                for k in range(n): pygame.draw.rect(self.screen, (80,60,40), (sx, sy+k*25, 40, 20), border_radius=3)
            elif p==2:
                w = n*25; sx = (SCREEN_WIDTH - w)//2; sy = 20
                for k in range(n): pygame.draw.rect(self.screen, (80,60,40), (sx+k*25, sy, 20, 40), border_radius=3)
            elif p==3:
                h = n*25; sy = (SCREEN_HEIGHT - h)//2; sx = 10
                for k in range(n): pygame.draw.rect(self.screen, (80,60,40), (sx, sy+k*25, 40, 20), border_radius=3)

    def draw_menu(self):
        self.screen.fill((20, 20, 25))
        t = self.big_font.render("DOMINÓ PRO IA", True, HIGHLIGHT)
        self.screen.blit(t, (SCREEN_WIDTH//2 - t.get_width()//2, 100))
        
        opts = [("1 vs 1", 2, False), ("4 Jugadores (FFA)", 4, False), ("2 vs 2 (Equipos)", 4, True)]
        mx, my = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]
        
        current_time = pygame.time.get_ticks()
        # FIX: Si hemos hecho clic hace menos de 300ms, ignorar clics en el menú
        input_blocked = (current_time - self.last_click_time < 300)

        for i, (txt, n, tm) in enumerate(opts):
            rect = pygame.Rect(SCREEN_WIDTH//2 - 150, 250 + i*90, 300, 70)
            col = (50, 100, 50) if rect.collidepoint(mx, my) else (40, 40, 50)
            pygame.draw.rect(self.screen, col, rect, border_radius=10)
            pygame.draw.rect(self.screen, (200,200,200), rect, 2, border_radius=10)
            
            surf = self.font.render(txt, True, (255,255,255))
            self.screen.blit(surf, (rect.centerx - surf.get_width()//2, rect.centery - surf.get_height()//2))
            
            # FIX: Solo procesar si el input no está bloqueado por el cooldown
            if click and rect.collidepoint(mx, my) and not input_blocked:
                self.game = DominoGame(n, tm)
                self.state = "PLAY"
                self.selected_tile_idx = None
                self.last_click_time = current_time # Actualizamos el tiempo para el nuevo juego
                pygame.time.delay(100)

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
                  
                if not self.game.game_over:
                    turn_txt = f"Turno: {'TÚ' if self.game.current_player==0 else f'BOT {self.game.current_player}'}"
                    self.screen.blit(self.font.render(turn_txt, True, (255,255,255)), (20, SCREEN_HEIGHT-100))
                    
                    if len(self.game.mesa) == 0:
                        start_info = getattr(self.game, 'start_reason', '')
                        st = self.font.render(start_info, True, HIGHLIGHT)
                        self.screen.blit(st, (20, 20))
                
                if not self.game.game_over:
                    turn = self.game.current_player
                    if turn == 0: # Humano
                        if not self.game.get_valid_moves(0):
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

                for e in pygame.event.get():
                    if e.type == pygame.QUIT: pygame.quit(); sys.exit()
                    
                    if e.type == pygame.MOUSEBUTTONDOWN:
                        if self.game.game_over:
                            # FIX: Registramos el tiempo exacto del clic para activar el cooldown
                            self.last_click_time = pygame.time.get_ticks()
                            self.state = "MENU" 
                            self.game = None
                        
                        elif self.game.current_player == 0:
                            mx, my = pygame.mouse.get_pos()
                            processed_click = False

                            for rect, idx, ficha in self.tile_rects:
                                if rect.collidepoint(mx, my):
                                    processed_click = True
                                    
                                    valid = self.game.get_valid_moves(0)
                                    possible_moves = [m for m in valid if m[0] == ficha]
                                    
                                    if not possible_moves:
                                        self.selected_tile_idx = None
                                    elif len(possible_moves) == 1:
                                        self.selected_tile_idx = idx 
                                        self.game.step(possible_moves[0])
                                        self.selected_tile_idx = None 
                                    else:
                                        self.selected_tile_idx = idx
                                        
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
                                    
                                    break

                pygame.display.flip()
                self.clock.tick(30)

if __name__ == "__main__":
    DominoGUI().run()