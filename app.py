# app.py
import streamlit as st
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# -------------------------
# Datos de portada
# -------------------------
titulo = "Búsqueda y Optimización"
actividad = "Actividad 1"
master = "Máster en IA para el sector de la Energía y las Infraestructuras"
autores = ["Elena Ruiz", "Cristina Jiménez", "Airam Betancor"]

# -------------------------
# Contenedor de portada
# -------------------------
if 'start_app' not in st.session_state:
    st.session_state.start_app = False

if not st.session_state.start_app:
    st.title(titulo)
    st.subheader(actividad)
    st.write(master)
    st.write("Autores: " + ", ".join(autores))

    st.button(
        "Continuar al algoritmo",
        on_click=lambda: st.session_state.update({'start_app': True})
    )
else:
    st.title("Algoritmo A*")
    st.write("Selecciona origen, destino y heurística")

    # --------------------------
    # Datos del grafo
    # --------------------------
    edges = [
        ("A","B","Verde",2,280), ("A","C","Naranja",5,563),
        ("B","A","Rojo",8,272), ("B","D","Verde",2,649),
        ("C","F","Verde",2,718), ("D","C","Verde",2,361),
        ("D","E","Naranja",5,323), ("E","H","Verde",2,340),
        ("E","D","Rojo",8,343), ("E","F","Verde",2,411),
        ("F","A","Naranja",5,371), ("F","E","Rojo",8,356),
        ("G","F","Verde",2,372), ("G","E","Naranja",5,532),
        ("H","D","Rojo",8,466), ("H","G","Rojo",8,727),
    ]

    G = nx.DiGraph()
    for o,d,color,cost,km in edges:
        G.add_edge(o, d, color=color, cost_state=cost, km=km)

    nodes = sorted(list(set([n for e in edges for n in e[:2]])))
    start = st.selectbox("Nodo inicio", nodes, index=nodes.index("A") if "A" in nodes else 0)
    goal = st.selectbox("Nodo destino", nodes, index=nodes.index("H") if "H" in nodes else -1)

    # --------------------------
    # Selección de heurística
    # --------------------------
    heur_option = st.selectbox(
        "Selecciona una heurística",
        ["Arco más corto × costo más barato", "Costo uniforme (h=0)"]
    )

    def h_informada(nodo):
        outgoing = [attrs["km"] for _, _, attrs in G.out_edges(nodo, data=True)]
        return min(outgoing) * 2 if outgoing else 0
    
    def h_nula(nodo):
        return 0
    
    if heur_option == "Costo uniforme (h=0)":
        h_fn = h_nula
        st.caption("Heurística nula: el algoritmo se comporta como Dijkstra.")
    else:
        h_fn = h_informada
        st.caption("Heurística informada: km mínimo saliente × 2.")

    # --------------------------
    # A* clásico con logging completo
    # --------------------------
    def a_star(graph, start, goal, heuristic_fn):
        """
        A* clásico completo:
        - Siempre expandimos el nodo de menor f.
        - Todos los hijos se agregan al OPEN inmediatamente al expandir el nodo padre.
        - El registro de expansiones refleja el orden de generación de hijos.
        """
        open_heap = []
        closed_set = set()
        counter = 0
    
        start_node = {
            "state": start,
            "g": 0,
            "h": heuristic_fn(start),
            "f": heuristic_fn(start),
            "parent": None
        }
    
        heapq.heappush(open_heap, (start_node["f"], counter, start_node))
        counter += 1
    
        expansions = []
        solution_node = None
    
        while open_heap:
            f_current, _, current = heapq.heappop(open_heap)
    
            # Saltar si ya está cerrado
            if current["state"] in closed_set:
                continue
    
            # Marcar como cerrado
            closed_set.add(current["state"])
    
            # Guardamos el nodo padre (expandido)
            expansions.append(current)
    
            # Condición de parada
            if current["state"] == goal:
                solution_node = current
                break
    
            # EXPANDIR TODOS LOS HIJOS DEL NODO ACTUAL
            children_nodes = []
            for _, neighbor, attrs in graph.out_edges(current["state"], data=True):
                g_new = current["g"] + attrs["km"] * attrs["cost_state"]
                h_new = heuristic_fn(neighbor)
                f_new = g_new + h_new
    
                child = {
                    "state": neighbor,
                    "g": g_new,
                    "h": h_new,
                    "f": f_new,
                    "parent": current
                }
    
                # Añadimos al registro en orden de generación
                children_nodes.append(child)
    
                # Todos los hijos van al OPEN (para selección A* según f)
                heapq.heappush(open_heap, (f_new, counter, child))
                counter += 1
    
            # Guardamos todos los hijos en el log inmediatamente después del padre
            expansions.extend(children_nodes)
    
        return solution_node, expansions

    
    # --------------------------
    # Dibujar árbol de expansión jerárquico
    # --------------------------
    def draw_expansion_tree(solution_node, expansions):
        G_tree = nx.DiGraph()
        labels = {}
        colors = {}
        id_map = {}

        for i, node in enumerate(expansions):
            node_id = f"{node['state']}_{i}"
            id_map[id(node)] = node_id
            G_tree.add_node(node_id)
            labels[node_id] = (
                f"{node['state']} ({i})\n"
                f"g={node['g']:.0f}, h={node['h']:.0f}, f={node['f']:.0f}"
            )
            colors[node_id] = "lightgray"

        for node in expansions:
            if node["parent"]:
                G_tree.add_edge(id_map[id(node["parent"])], id_map[id(node)])

        current = solution_node
        while current:
            colors[id_map[id(current)]] = "lightgreen"
            current = current["parent"]

        parent_children = defaultdict(list)
        for node in expansions:
            if node["parent"]:
                parent_children[id_map[id(node["parent"])]].append(id_map[id(node)])

        pos = {}
        y_gap = 2.5
        x_gap = 2.5

        def assign_pos(node, x_center, lvl):
            children = parent_children.get(node, [])
            n = len(children)
            pos[node] = (x_center, -lvl * y_gap)
            if n == 0:
                return
            width = (n - 1) * x_gap
            for i, child in enumerate(children):
                x_child = x_center - width / 2 + i * x_gap
                assign_pos(child, x_child, lvl + 1)

        root_id = id_map[id(expansions[0])]
        assign_pos(root_id, 0, 0)

        fig, ax = plt.subplots(figsize=(18, 10))
        nx.draw_networkx_edges(G_tree, pos, arrows=True, arrowstyle='-|>', arrowsize=10, ax=ax)
        for node, (x, y) in pos.items():
            ax.text(
                x, y,
                labels[node],
                ha='center', va='center',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=colors[node], edgecolor="black")
            )
        ax.axis("off")
        st.pyplot(fig)

    # --------------------------
    # Ejecutar A* y mostrar resultados
    # --------------------------
    if st.button("Ejecutar A*"):
        solution, expansions = a_star(G, start, goal, h_fn)

        if solution:
            st.subheader("Camino óptimo")
            path = []
            cur = solution
            while cur:
                path.append(cur)
                cur = cur["parent"]
            path = path[::-1]

            table = [
                {"node": n["state"], "g": n["g"], "h": n["h"], "f": n["f"]}
                for n in path
            ]
            st.table(pd.DataFrame(table).style.format({"g": "{:.2f}", "h": "{:.2f}", "f": "{:.2f}"}))

            st.subheader("Grafo final")
            pos_fixed = {
                "A": (0, 2.7), "B": (1, 3), "C": (2, 3), "F": (0.2, 1.7),
                "D": (2.2, 2), "E": (1, 1), "G": (0, 0), "H": (2.1, 0)
            }
            fig, ax = plt.subplots(figsize=(7, 6))
            nx.draw_networkx_nodes(G, pos_fixed, node_size=800, node_color="white", edgecolors="black", ax=ax)
            nx.draw_networkx_labels(G, pos_fixed, font_weight="bold", ax=ax)

            edge_colors = []
            for u, v, attrs in G.edges(data=True):
                col = attrs["color"].lower()
                if col.startswith("v"):
                    edge_colors.append("green")
                elif col.startswith("n"):
                    edge_colors.append("orange")
                else:
                    edge_colors.append("red")

            nx.draw_networkx_edges(G, pos_fixed, edge_color=edge_colors, ax=ax)
            path_edges = list(zip([n["state"] for n in path[:-1]], [n["state"] for n in path[1:]]))
            nx.draw_networkx_edges(
                G, pos_fixed, edgelist=path_edges,
                edge_color="blue", width=3, arrows=True, arrowstyle='-|>'
            )
            edge_labels = {(u, v): f"{attrs['km']}km/{attrs['cost_state']}" for u, v, attrs in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos_fixed, edge_labels=edge_labels, ax=ax)
            ax.axis("off")
            st.pyplot(fig)

            st.subheader("Árbol de expansión")
            st.write("En verde se muestran los nodos en el camino final.")
            draw_expansion_tree(solution, expansions)
