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