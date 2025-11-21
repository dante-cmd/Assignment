import threading
import math
import random
import copy
from typing import List, Optional

# ========================
# Node con soporte para virtual loss y lock local
# ========================
class Node:
    def __init__(self, move=None, parent: 'Node' = None):
        self.move = move
        self.parent = parent
        self.children: List[Node] = []
        self.w = 0.0           # suma de recompensas
        self.visits = 0        # número real de visitas
        self.virtual_loss = 0  # contador temporal de virtual loss
        self.lock = threading.RLock()  # Lock por nodo (reentrante)
        self.untried_actions = []  # solo usado en root inicialmente

    def uct_value(self, c_param: float = math.sqrt(2)):
        if self.visits + self.virtual_loss == 0:
            return float('inf')
        Q = self.w / (self.visits + self.virtual_loss)
        U = c_param * math.sqrt(math.log(self.parent.visits + self.parent.virtual_loss) / (self.visits + self.virtual_loss))
        return Q + U

    def select_child(self):
        with self.lock:
            return max(self.children, key=lambda c: c.uct_value())

    def add_child(self, move, untried_actions):
        child = Node(move=move, parent=self)
        child.untried_actions = untried_actions[:]
        self.children.append(child)
        return child

    def update(self, reward: float):
        with self.lock:
            self.w += reward
            self.visits += 1

    # Aplicar virtual loss (se llama al descender)
    def apply_virtual_loss(self):
        with self.lock:
            self.virtual_loss += 1

    # Revertir virtual loss y actualizar con resultado real
    def revert_virtual_loss_and_update(self, reward: float):
        with self.lock:
            self.virtual_loss -= 1
            self.w += reward
            self.visits += 1


# ========================
# MCTS Paralelo en Árbol Único con Virtual Loss
# ========================
def tree_parallel_mcts(
    root_state: RoomLog,
    available_actions: list,
    num_iterations: int = 5000,
    c_param: float = math.sqrt(2),
    num_threads: int = 12
):
    # Crear nodo raíz
    root = Node(move=None)
    root.untried_actions = available_actions[:]

    def mcts_worker():
        state = root_state.clone()
        path: List[Node] = []

        for _ in range(10000):  # cada hilo corre hasta que se alcance el total
            node = root
            state.idx_item = root_state.idx_item
            state.roomlog = copy.deepcopy(root_state.roomlog)

            path.clear()

            # === 1. Selection + Expansion ===
            while True:
                node.lock.acquire()
                if node.untried_actions:
                    # Expansion
                    action = random.choice(node.untried_actions)
                    node.untried_actions.remove(action)
                    reward = state.step(action)
                    child = node.add_child(action, state.get_available_actions())
                    node.lock.release()

                    node = child
                    path.append(node)
                    node.apply_virtual_loss()
                    path.append(node)
                    break
                elif node.children:
                    # Selection con virtual loss
                    node.apply_virtual_loss()
                    path.append(node)
                    child = node.select_child()
                    node.lock.release()

                    reward = state.step(child.move)
                    node = child
                else:
                    # Nodo hoja terminal
                    node.lock.release()
                    break

            # === 2. Simulation (rollout aleatorio) ===
            rewards = []
            while not state.is_terminal():
                actions = state.get_available_actions()
                if not actions:
                    break
                action = random.choice(actions)
                r = state.step(action)
                if r is not None:
                    rewards.append(r)

            total_reward = sum(rewards) / max(state.n_items, 1) if rewards else 0.0

            # === 3. Backpropagation (revertir virtual loss + update real) ===
            for n in reversed(path):
                if n.parent is not None:  # no actualizamos root con virtual loss
                    n.revert_virtual_loss_and_update(total_reward)

            # Detener si se alcanzó el número total de simulaciones
            if root.visits >= num_iterations:
                return

    # Lanzar hilos
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=mcts_worker, daemon=True)
        t.start()
        threads.append(t)

    # Esperar a que todos terminen o se alcance el número de iteraciones
    for t in threads:
        t.join()

    # Seleccionar mejor hijo
    if not root.children:
        return None, root

    best_child = max(root.children, key=lambda c: c.visits)
    print(f"[MCTS] Root visits: {root.visits}, Best move visits: {best_child.visits}, Q: {best_child.w/best_child.visits:.3f}")
    return best_child.move, root


def run_mcts(periodo, sede, data_path, room_log_path, items_path, items_predict_path,
             iter_max: int = 10000, num_threads: int = 16):
    state = RoomLog(periodo, sede, data_path, room_log_path, items_path, items_predict_path)
    aulas = []

    print(f"Starting MCTS Tree-Parallel (threads={num_threads}, iters={iter_max})")

    while state.idx_item < len(state.items):
        print(f"\n--- Item {state.idx_item + 1}/{len(state.items)} ---")
        available_actions = state.get_available_actions()

        if not available_actions:
            result = state.items[state.idx_item].copy()
            result['ASSIGNMENTS'] = {'AULA': None, 'AFORO': None}
            state.idx_item += 1
        else:
            move, root = tree_parallel_mcts(
                root_state=state,
                available_actions=available_actions,
                num_iterations=iter_max,
                num_threads=num_threads
            )
            if move is None:
                move = random.choice(available_actions)

            aula = state.aulas[move]
            aforo = state.aforos[move]
            result = state.items[state.idx_item].copy()
            result['ASSIGNMENTS'] = {'AULA': aula, 'AFORO': aforo}

            state.step(move)  # aplicar movimiento real

        aulas.append(result)

    # Guardar resultados
    df = pd.DataFrame(aulas)
    output_path = PATH_ASSIGNMENT / f'{periodo}'
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path / f'asignacion_{periodo}_{sede}.xlsx', index=False)
    print("Asignación completada y guardada.")