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
        st.caption("Heurística subestimada basada en el arco saliente más corto, multiplicando por el costo más barato.")

    # --------------------------
    # Función A* simple
    # --------------------------
    def a_star(graph, start, goal, heuristic_fn):
        """
        A* simple para visualización de árbol de expansión.
        Permite nodos repetidos y se detiene solo cuando
        el objetivo es el nodo con f mínima en OPEN.
        """
        open_heap = []
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

            # Guardamos la expansión
            expansions.append(current)

            # Condición de parada correcta
            if current["state"] == goal:
                if not open_heap or f_current <= open_heap[0][0]:
                    solution_node = current
                    break

            # # Expandir hijos
            # for _, neighbor, attrs in graph.out_edges(current["state"], data=True):
            #     g_new = current["g"] + attrs["km"] * attrs["cost_state"]
            #     h_new = heuristic_fn(neighbor)
            #     child = {
            #         "state": neighbor,
            #         "g": g_new,
            #         "h": h_new,
            #         "f": g_new + h_new,
            #         "parent": current
            #     }
            #     heapq.heappush(open_heap, (child["f"], counter, child))
            #     counter += 1
            # Expandir hijos
            # Solo agregamos hijos al log, el padre ya está incluido
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
                heapq.heappush(open_heap, (f_new, counter, child))
                counter += 1
                children_nodes.append(child)
            
            # Añadir hijos al log
            expansions.extend(children_nodes)

                

        return solution_node, expansions

    # --------------------------
    # Dibujar árbol de expansión
    # --------------------------
    def draw_expansion_tree(solution_node, expansions):
        import networkx as nx
        import matplotlib.pyplot as plt
        from collections import defaultdict
    
        G_tree = nx.DiGraph()
        labels = {}
        colors = {}
        id_map = {}
    
        # Crear nodos únicos y asignar label con iteración
        for i, node in enumerate(expansions):
            node_id = f"{node['state']}_{i}"
            id_map[id(node)] = node_id
            G_tree.add_node(node_id)
            labels[node_id] = f"{node['state']} ({i})\ng={node['g']:.0f}\nh={node['h']:.0f}\nf={node['f']:.0f}"
            colors[node_id] = "lightgray"
    
        # Crear aristas padre -> hijo
        for node in expansions:
            if node["parent"]:
                G_tree.add_edge(id_map[id(node["parent"])], id_map[id(node)])
    
        # Marcar camino solución
        current = solution_node
        while current:
            colors[id_map[id(current)]] = "lightgreen"
            current = current["parent"]
    
        # ----------------------
        # Posición manual jerárquica
        # ----------------------
        levels = defaultdict(list)
        root = id_map[id(expansions[0])]
        queue = [(root, 0)]
        visited = set()
        parent_children = defaultdict(list)
    
        # BFS para niveles y asignar hijos
        for node in expansions:
            if node["parent"]:
                parent_children[id_map[id(node["parent"])]].append(id_map[id(node)])
    
        y_gap = 3
        x_gap = 3
        pos = {}
    
        def assign_pos(node, x_center, y_level):
            children = parent_children.get(node, [])
            n = len(children)
            if n == 0:
                pos[node] = (x_center, -y_level * y_gap)
                return
            # repartir horizontalmente
            width = (n-1) * x_gap
            for i, child in enumerate(children):
                x_child = x_center - width/2 + i * x_gap
                pos[child] = (x_child, - (y_level+1) * y_gap)
                assign_pos(child, x_child, y_level+1)
    
        pos[root] = (0, 0)
        assign_pos(root, 0, 0)
    
        # ----------------------
        # Dibujar
        # ----------------------
        fig, ax = plt.subplots(figsize=(18, 10))
        nx.draw_networkx_edges(G_tree, pos, arrows=True, arrowstyle='-|>', arrowsize=12, ax=ax)
        for n, (x, y) in pos.items():
            ax.text(x, y, labels[n], ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=colors[n], edgecolor="black"))
        ax.axis("off")
        st.pyplot(fig)

    


    

    # --------------------------
    # Ejecutar A* y mostrar resultados
    # --------------------------
    if st.button("Ejecutar A*"):
        solution, expansions = a_star(G, start, goal, h_fn)

        if solution:
            # Camino óptimo
            st.subheader("Camino óptimo")
            path = []
            current = solution
            while current:
                path.append(current)
                current = current["parent"]
            path = path[::-1]

            final_rows = [{"node": n["state"], "g": n["g"], "h": n["h"], "f": n["f"]} for n in path]
            st.table(pd.DataFrame(final_rows).style.format({"g": "{:.2f}", "h": "{:.2f}", "f": "{:.2f}"}))

            # Grafo final
            st.subheader("Grafo")
            st.write("En color azul se muestra el camino escogido:")

            pos_fixed = { "A": (0, 2.7), "B": (1, 3), "C": (2, 3), "F": (0.2, 1.7), 
                          "D": (2.2, 2), "E": (1, 1), "G": (0, 0), "H": (2.1, 0) }
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

            if len(path)>1:
                path_edges = list(zip([n["state"] for n in path[:-1]], [n["state"] for n in path[1:]]))
                nx.draw_networkx_edges(G, pos_fixed, edgelist=path_edges, edge_color='blue', width=4.0,
                                       arrows=True, arrowstyle='-|>', arrowsize=16,
                                       connectionstyle='arc3,rad=0.2',
                                       ax=ax, min_source_margin=15, min_target_margin=15)

            nx.draw_networkx_edges(G, pos_fixed, edge_color=edge_colors, width=widths,
                                   arrows=True, arrowstyle='-|>', arrowsize=10,
                                   connectionstyle='arc3,rad=0.2', ax=ax,
                                   min_source_margin=15, min_target_margin=15)
            edge_labels = {(u,v): f"{attrs['km']}km/{attrs['cost_state']}" for u,v,attrs in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos_fixed, edge_labels=edge_labels, font_size=8, ax=ax)
            ax.axis('off')
            st.pyplot(fig)

            # Árbol de expansión
            st.subheader("Árbol de expansión")
            st.write("En verde se muestran los nodos finales escogidos")
            draw_expansion_tree(solution, expansions)
