# app.py
import streamlit as st
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

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

    # Botón para continuar al algoritmo
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

    # --------------------------
    # Construir grafo
    # --------------------------
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

    def heuristic(node, graph, closed):
        outgoing = [attrs["km"] for _, n, attrs in graph.out_edges(node, data=True) if n not in closed]
        if not outgoing: 
            return 0
        return min(outgoing) * 2

    def heuristic_wrapper(node, graph, closed, goal):
        if node == goal:
            return 0
        if heur_option == "Costo uniforme (h=0)":
            return 0
        else:
            return heuristic(node, graph, closed)

    if heur_option == "Costo uniforme (h=0)":
        st.caption("Heurística nula: el algoritmo se comporta como Dijkstra.")
    else:
        st.caption("Heurística subestimada basada en el arco saliente más corto, multiplicando por el costo más barato.")

    # --------------------------
    # Algoritmo A* completo
    # --------------------------
    def a_star_full(graph, start, goal):
        open_heap = []
        heapq.heappush(open_heap, (0, start))
        came_from = {}
        closed = set()
        g = {start: 0}
        h = {start: heuristic_wrapper(start, graph, closed, goal)}
        f = {start: g[start] + h[start]}
        expansion_log = []
        step = 1
        all_nodes = {start: None}  # nodo -> padre

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue

            neighbors = [n for _, n, _ in graph.out_edges(current, data=True) if n not in closed]
            hcur = h[current]
            fcur = f[current]
            open_nodes = [n for _, n in open_heap]
            expansion_log.append((step, current, g[current], hcur, fcur, neighbors, open_nodes, list(closed)))
            step += 1

            if current == goal:
                if open_heap and open_heap[0][0] < f[current]:
                    closed.add(current)
                    continue
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path = path[::-1]
                return {
                    "path": path,
                    "log": expansion_log,
                    "g": g,
                    "h": h,
                    "f": f,
                    "came_from": came_from,
                    "all_nodes": all_nodes
                }

            closed.add(current)

            for _, neighbor, attrs in graph.out_edges(current, data=True):
                if neighbor in closed:
                    continue
                tentative_g = g[current] + attrs['km'] * attrs['cost_state']
                tentative_h = heuristic_wrapper(neighbor, graph, closed, goal)
                tentative_f = tentative_g + tentative_h
                if neighbor not in g or tentative_g < g[neighbor]:
                    came_from[neighbor] = current
                    g[neighbor] = tentative_g
                    h[neighbor] = tentative_h
                    f[neighbor] = tentative_f
                    heapq.heappush(open_heap, (tentative_f, neighbor))
                if neighbor not in all_nodes:
                    all_nodes[neighbor] = current

        return {"path": None, "log": expansion_log, "g": g, "h": h, "f": f, "came_from": came_from, "all_nodes": all_nodes}

    # --------------------------
    # Dibujar árbol de expansión
    # --------------------------
    def draw_decision_tree(solution_path, expansion_log, g_vals, h_vals, f_vals):
        import networkx as nx
        import matplotlib.pyplot as plt
        from collections import defaultdict
    
        G_tree = nx.DiGraph()
        node_labels = {}
        node_colors = {}
        children = defaultdict(list)
    
        # --- 1. Crear nodos por expansión ---
        exp_nodes = {}
    
        for step, current, g, h, f, neighbors, *_ in expansion_log:
            node_id = f"{current}_{step}"
            exp_nodes[(current, step)] = node_id
            G_tree.add_node(node_id)
    
            node_labels[node_id] = (
                f"{current} ({step})\n"
                f"g={g:.0f}\n"
                f"h={h:.0f}\n"
                f"f={f:.0f}"
            )
    
            node_colors[node_id] = (
                "lightgreen" if solution_path and current in solution_path else "lightgray"
            )
    
        # --- 2. Crear aristas padre → hijos ---
        for i, (step, current, g, h, f, neighbors, *_ ) in enumerate(expansion_log):
            parent_id = exp_nodes[(current, step)]
    
            for neigh in neighbors:
                for s2, cur2, *_ in expansion_log[i+1:]:
                    if cur2 == neigh:
                        child_id = exp_nodes[(cur2, s2)]
                        G_tree.add_edge(parent_id, child_id)
                        children[parent_id].append(child_id)
                        break
    
        # --- 3. Calcular niveles (BFS desde raíces reales) ---
        roots = [n for n in G_tree.nodes() if G_tree.in_degree(n) == 0]
    
        level = {}
        queue = []
    
        for r in roots:
            level[r] = 0
            queue.append(r)
    
        while queue:
            parent = queue.pop(0)
            for child in children.get(parent, []):
                if child not in level:
                    level[child] = level[parent] + 1
                    queue.append(child)
    
        # --- 4. Asignar posiciones (TODOS los nodos) ---
        pos = {}
        level_nodes = defaultdict(list)
    
        for node, lvl in level.items():
            level_nodes[lvl].append(node)
    
        y_gap = 2.5
        x_gap = 2.8
    
        for lvl, nodes in level_nodes.items():
            for i, node in enumerate(nodes):
                pos[node] = (i * x_gap, -lvl * y_gap)
    
        # 🔒 Seguridad: nodos sin nivel (no debería haber, pero por si acaso)
        missing = set(G_tree.nodes()) - set(pos.keys())
        for i, node in enumerate(missing):
            pos[node] = (i * x_gap, -(max(level.values(), default=0) + 1) * y_gap)
    
        # --- 5. Dibujar ---
        fig, ax = plt.subplots(figsize=(18, 10))
    
        nx.draw_networkx_edges(
            G_tree, pos,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=12,
            ax=ax
        )
    
        for node, (x, y) in pos.items():
            ax.text(
                x, y,
                node_labels[node],
                ha='center', va='center',
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor=node_colors[node],
                    edgecolor='black'
                )
            )
    
        ax.axis('off')
        st.pyplot(fig)



       

    # --------------------------
    # Ejecutar A* y mostrar resultados
    # --------------------------
    if st.button("Ejecutar A*"):
        result = a_star_full(G, start, goal)

        # Camino óptimo
        if result["path"]:
            st.subheader("Camino óptimo")
            final_rows = [{"node": n, "g": result["g"][n], "h": result["h"][n], "f": result["f"][n]} for n in result["path"]]
            st.table(pd.DataFrame(final_rows).style.format({"g": "{:.2f}", "h": "{:.2f}", "f": "{:.2f}"}))

        # Grafo final
        st.subheader("Grafo")
        st.write("En color azul se muestra el camino escogido:")
        pos_fixed = { "A": (0, 2.7), "B": (1, 3), "C": (2, 3), "F": (0.2, 1.7), "D": (2.2, 2),
                      "E": (1, 1), "G": (0, 0), "H": (2.1, 0) }
        fig, ax = plt.subplots(figsize=(7,6))
        nx.draw_networkx_nodes(G, pos_fixed, node_size=800, node_color="white", edgecolors="black", linewidths=2, ax=ax)
        nx.draw_networkx_labels(G, pos_fixed, font_weight='bold', ax=ax)

        edge_colors = []
        widths = []
        for u,v,attrs in G.edges(data=True):
            col = attrs.get('color','gray').lower()
            if col.startswith('v'): edge_colors.append('green')
            elif col.startswith('n'): edge_colors.append('orange')
            elif col.startswith('r'): edge_colors.append('red')
            else: edge_colors.append('gray')
            widths.append(2.0)

        if result["path"] and len(result["path"])>1:
            path_edges = list(zip(result["path"][:-1], result["path"][1:]))
            nx.draw_networkx_edges(
                G, pos_fixed, edgelist=path_edges, edge_color='blue', width=4.0,
                arrows=True, arrowstyle='-|>', arrowsize=16,
                connectionstyle='arc3,rad=0.2',
                ax=ax, min_source_margin=15, min_target_margin=15
            )

        nx.draw_networkx_edges(G, pos_fixed, edge_color=edge_colors, width=widths,
                               arrows=True, arrowstyle='-|>', arrowsize=10,
                               connectionstyle='arc3,rad=0.2', ax=ax,
                               min_source_margin=15, min_target_margin=15)
        edge_labels = {(u,v): f"{attrs['km']}km/{attrs['cost_state']}" for u,v,attrs in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos_fixed, edge_labels=edge_labels, font_size=8, ax=ax)
        ax.axis('off')
        st.pyplot(fig)

        # Árbol de expansión
        if result["path"]:
            st.subheader("Árbol de expansión")
            st.write("En verde se muestran los nodos finales escogidos")
            draw_decision_tree(
                solution_path=result["path"],
                expansion_log=result["log"],
                g_vals=result["g"],
                h_vals=result["h"],
                f_vals=result["f"]
            )

