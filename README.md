# Projet IA Générative – Analyse sémantique et recommandation de métiers
## Présentation du projet

Ce projet a été réalisé dans le cadre du Projet IA Générative du mastère Data Engineering et IA.
L’objectif est de concevoir une application capable d’analyser sémantiquement le profil d’un utilisateur afin de lui recommander des métiers adaptés dans le domaine de la Data et de l’Intelligence Artificielle, tout en proposant un plan de progression personnalisé et une biographie professionnelle.

L’application repose sur un formulaire interactif permettant de collecter des informations sur les compétences, les expériences, les projets et les centres d’intérêt de l’utilisateur.

# Fonctionnalités principales

Analyse sémantique du profil utilisateur à partir de texte libre

Génération d’embeddings à l’aide de SBERT

Calcul de similarités cosinus et scoring par blocs de compétences

Recommandation des métiers les plus proches du profil

Génération d’un plan de progression personnalisé

Génération d’une biographie professionnelle synthétique

Visualisation des résultats via une interface Streamlit

# Architecture du projet

Le projet est organisé autour d’un pipeline structuré comprenant :

La collecte des données utilisateur via l’interface Streamlit

Le prétraitement du texte (nettoyage et normalisation)

L’analyse sémantique avec SBERT

Le calcul des scores de similarité et la recommandation de métiers

L’enrichissement des résultats à l’aide d’une IA générative

L’affichage des résultats sous forme de graphiques et de textes explicatifs

# Technologies utilisées

Python – langage principal

Streamlit – interface utilisateur

SBERT (Sentence-BERT) – embeddings sémantiques

Google Gemini – génération du plan de progression et de la bio

matplotlib – visualisation des résultats

JSON – stockage des référentiels et des résultats

GitHub – gestion de version

Microsoft Teams – communication et suivi du projet

# Structure des fichiers

app.py : interface Streamlit et orchestration du pipeline

preprocessing.py : nettoyage et préparation du texte

embeddings.py : génération des embeddings SBERT

similarity.py : calcul des similarités et scoring

generator.py : génération IA (enrichissement, plan, bio)

data/competencies.json : blocs de compétences

data/jobs.json : référentiel métiers

outputs/user_results.json : historique des résultats utilisateurs

cache.json : cache des réponses Gemini

requirements.txt : dépendances Python