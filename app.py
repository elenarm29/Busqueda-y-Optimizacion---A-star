# app.py
import streamlit as st
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

import plotly.graph_objects as go


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
        ["Arco más corto × costo más barato", "Costo uniforme (h=0)", "Heurística perfecta (distancia real)"]
    )



  

    def h_informada(nodo):
        if nodo == goal:
            return 0
        outgoing = [attrs["km"] for _, _, attrs in G.out_edges(nodo, data=True)]
        return min(outgoing) * 2 if outgoing else 0
    
    def h_nula(nodo):
        return 0
    
    def h_perfecta(nodo):
        if nodo == goal:
            return 0
        try:
            # Calculamos el coste real considerando km * cost_state
            return nx.shortest_path_length(G, source=nodo, target=goal, weight=lambda u, v, d: d['km'] * d['cost_state'])
        except nx.NetworkXNoPath:
            return float('inf') # Si no hay camino, el coste es infinito

    
    if heur_option == "Costo uniforme (h=0)":
        h_fn = h_nula
        st.caption("Heurística nula: el algoritmo se comporta como Dijkstra.")
    elif heur_option == "Heurística perfecta (distancia real)":
        h_fn = h_perfecta
        st.caption("Heurística ideal: Conoce el coste exacto al destino. A* irá directo.")
    else:
        h_fn = h_informada
        st.caption("Heurística subestimada basada en el arco saliente más corto, multiplicando por el costo más barato.")
    
    # --------------------------
    # Función A* clásica
    # --------------------------
    def a_star(graph, start, goal, heuristic_fn):
        open_heap = []
        counter = 0    
        expansions = [] 
        
        # Este contador marcará el orden de APARICIÓN de cada nodo
        node_creation_counter = 0

        # Nodo raíz (es el primero en aparecer: 0)
        start_node = {
            "state": start,
            "g": 0,
            "h": heuristic_fn(start),
            "f": heuristic_fn(start),
            "parent": None,
            "iteration": node_creation_counter, 
            "children": [] 
        }
        node_creation_counter += 1

        heapq.heappush(open_heap, (start_node["f"], counter, start_node))
        counter += 1
        solution_node = None

        while open_heap:
            f_val, _, current = heapq.heappop(open_heap)
            expansions.append(current)

            if current["state"] == goal:
                solution_node = current
                break

            neighbors = sorted([v for _, v, _ in graph.out_edges(current["state"], data=True)])
            for neighbor in neighbors:
                attrs = graph.get_edge_data(current["state"], neighbor)
                g_new = current["g"] + attrs["km"] * attrs["cost_state"]
                h_new = 0 if neighbor == goal else heuristic_fn(neighbor)
                f_new = g_new + h_new

                child = {
                    "state": neighbor,
                    "g": g_new,
                    "h": h_new,
                    "f": f_new,
                    "parent": current,
                    "iteration": node_creation_counter, # Se le asigna su número al nacer
                    "children": []
                }
                node_creation_counter += 1 # Aumentamos para el siguiente
                
                current["children"].append(child)
                heapq.heappush(open_heap, (f_new, counter, child))
                counter += 1

        return solution_node, expansions

    # --------------------------
    # Árbol de expansión
    # --------------------------
    def draw_expansion_tree(solution_node, expansions):
        # 1. Recopilar todos los nodos generados
        all_nodes = []
        for e in expansions:
            if e not in all_nodes: all_nodes.append(e)
            for child in e["children"]:
                if child not in all_nodes: all_nodes.append(child)

        # 2. Mapeo de relaciones y datos
        parent_children = defaultdict(list)
        node_data = {}
        for i, node in enumerate(all_nodes):
            n_id = f"{node['state']}_{i}"
            node_data[n_id] = node
            if node["parent"]:
                for prev_id, prev_node in node_data.items():
                    if prev_node == node["parent"]:
                        parent_children[prev_id].append(n_id)
                        break

        pos = {}
        # Variable global para llevar el conteo de la X actual en las hojas
        # Esto asegura que el árbol se expanda hacia la derecha
        current_x_cursor = [0.0] 
        sep_x = 4.5  # Espacio horizontal entre nodos
        sep_y = 0.1  # Espacio vertical entre niveles

        def layout_tree(n_id, level):
            children = sorted(parent_children.get(n_id, []))
            
            if not children:
                # Si es una hoja, le asignamos la siguiente posición X disponible
                x = current_x_cursor[0]
                current_x_cursor[0] += sep_x
            else:
                # Si tiene hijos, primero posicionamos a los hijos
                child_x_positions = []
                for child in children:
                    child_x_positions.append(layout_tree(child, level + 1))
                
                # EL TRUCO: El padre se coloca en el centro exacto de sus hijos
                x = sum(child_x_positions) / len(child_x_positions)
            
            pos[n_id] = (x, -level * sep_y)
            return x

        # Ejecutar el posicionamiento desde la raíz
        if node_data:
            root_id = list(node_data.keys())[0]
            layout_tree(root_id, 0)

        # --- A partir de aquí, el código de dibujo de Plotly es el mismo ---
        # 3. Preparar elementos de Plotly
        edge_x, edge_y = [], []
        for p_id, children in parent_children.items():
            for c_id in children:
                edge_x += [pos[p_id][0], pos[c_id][0], None]
                edge_y += [pos[p_id][1], pos[c_id][1], None]

        node_x, node_y, node_labels, node_color, node_hover = [], [], [], [], []
        
        sol_path_nodes = []
        curr = solution_node
        while curr:
            sol_path_nodes.append(curr)
            curr = curr["parent"]

        for n_id, (x, y) in pos.items():
            node = node_data[n_id]
            node_x.append(x)
            node_y.append(y)
            display_text = (f"<b>{node['state']}</b> ({node['iteration']})<br>"
                            f"g:{node['g']:.0f}<br>h:{node['h']:.0f}<br>f:{node['f']:.0f}")
            node_labels.append(display_text)
            
            if node in sol_path_nodes: color = "#90ee90" 
            elif node in expansions: color = "#e0e0e0"   
            else: color = "#ffffff"                      
            node_color.append(color)
            node_hover.append(f"Nodo {node['state']}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#555'),
                                 hoverinfo='none', mode='lines'))

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_labels,
            textposition="middle center",
            textfont=dict(size=9, color='black'),
            hoverinfo='text',
            hovertext=node_hover,
            marker=dict(symbol='square', size=55, color=node_color, line=dict(width=2, color='black'))
        ))

        fig.update_layout(
            title="El gráfico es interactivo (arriba a la derecha de puede ampliar)",
            showlegend=False,
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=900,
            dragmode='pan'
        )

        st.plotly_chart(fig, use_container_width=True)  
    

    # --------------------------
    # Ejecutar
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
            # --------------------------
            # Grafo Visual con resaltado
            # --------------------------
            st.subheader("Grafo de rutas")
            st.write("El camino óptimo se resalta en azul (nodos y aristas) sobre la red original:")
            
            pos_fixed = { "A": (0, 2.7), "B": (1, 3), "C": (2, 3), "F": (0.2, 1.7), 
                          "D": (2.2, 2), "E": (1, 1), "G": (0, 0), "H": (2.1, 0) }
            
            # Identificar elementos del camino
            path_nodes = [n["state"] for n in path]
            path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 1. Dibujar aristas que NO son del camino (capa inferior)
            for u, v, attrs in G.edges(data=True):
                if (u, v) not in path_edges:
                    col = attrs.get('color','gray').lower()
                    c = 'green' if col.startswith('v') else ('orange' if col.startswith('n') else ('red' if col.startswith('r') else 'gray'))
                    nx.draw_networkx_edges(G, pos_fixed, edgelist=[(u,v)], edge_color=c, width=2.0,
                                           arrows=True, arrowstyle='-|>', arrowsize=10,
                                           connectionstyle='arc3,rad=0.2', ax=ax,
                                           min_source_margin=15, min_target_margin=15)

            # 2. Dibujar el camino óptimo (capa superior)
            if path_edges:
                nx.draw_networkx_edges(G, pos_fixed, edgelist=path_edges, edge_color='blue', width=3.0,
                                       arrows=True, arrowstyle='-|>', arrowsize=18,
                                       connectionstyle='arc3,rad=0.2', ax=ax,
                                       min_source_margin=15, min_target_margin=15)

            # 3. Dibujar nodos (Azul si están en el camino, blanco si no)
            node_colors = ['white' if n in path_nodes else 'white' for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos_fixed, node_size=800, node_color=node_colors, 
                                   edgecolors="black", linewidths=1, ax=ax)

            # 4. Etiquetas de los nodos (Texto blanco sobre azul para legibilidad)
            for node, (x, y) in pos_fixed.items():
                t_col = 'black' if node in path_nodes else 'black'
                ax.text(x, y, node, fontsize=11, fontweight='bold', va='center', ha='center', color=t_col)

            # Etiquetas de aristas
            edge_labels = {(u,v): f"{attrs['km']}km/{attrs['cost_state']}" for u,v,attrs in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos_fixed, edge_labels=edge_labels, font_size=7, ax=ax, label_pos=0.3)
            
            ax.axis('off')
            st.pyplot(fig)

            # Árbol de expansión
            st.subheader("Árbol de expansión")
            st.write("En verde se muestran los nodos finales escogidos. En gris, los nodos expandidos.")
            draw_expansion_tree(solution, expansions)
