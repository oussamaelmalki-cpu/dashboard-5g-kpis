# 📡 Dashboard 5G Privé - Supervision et Détection

👉 **[Essayez l'application en direct ici](https://dashboard-5g-kpis-wf3k2jm8uktzv7gfgq4alv.streamlit.app)**
Ce projet est une solution logicielle complète développée en **Python**. Il permet de simuler les indicateurs de performance (KPIs) d'un réseau 5G privé, d'analyser ces données via un moteur statistique avancé, et de les afficher sur une interface web interactive.

## 🎯 Objectif du Projet
Fournir aux opérateurs télécoms un outil visuel capable de **détecter les signes avant-coureurs d'une dégradation** avant que la panne ne survienne.

## 🏗️ Architecture du Projet
```mermaid
graph TD
    A[Simulateur de KPIs] -->|Données .csv| B(Prétraitement)
    B -->|Fenêtres temporelles| C{Moteur d'Analyse}
    C -->|Calcul de l'Indice| D[Indice d'Instabilité]
    D -->|Comparaison au Seuil| E[Tableau de Bord Streamlit]
    E -->|Visualisation| F((Opérateur))
