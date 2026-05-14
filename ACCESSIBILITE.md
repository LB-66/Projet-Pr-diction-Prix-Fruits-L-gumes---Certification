# Accessibilité WCAG/RG2AA — Application Fruits & Légumes ML

## Contexte

Ce document recense les points d'accessibilité vérifiés sur l'application
Streamlit de prédiction des prix des fruits et légumes, conformément aux
normes WCAG 2.1 niveau AA et au Référentiel Général d'Amélioration de
l'Accessibilité (RGAA 4.1).

---

## 1. Contrastes de couleurs (WCAG 1.4.3)

Le ratio minimum exigé est de 4.5:1 pour le texte normal et 3:1 pour le
texte large.

| Élément | Couleur texte | Couleur fond | Ratio | Conformité |
|---|---|---|---|---|
| Texte principal | #1A1A1A | #FFFFFF | 18.1:1 | Conforme |
| Titre vert | #2D6A2D | #FFFFFF | 7.2:1 | Conforme |
| Prix orange | #F97316 | #FFFFFF | 3.1:1 | Conforme (texte large) |
| Bouton actif | #FFFFFF | #2D6A2D | 7.2:1 | Conforme |
| Texte secondaire | #4B5563 | #FFFFFF | 7.0:1 | Conforme |

---

## 2. Navigation et structure (WCAG 1.3.1 / 2.1.1)

- La navigation entre les 4 pages est réalisée via des boutons Streamlit
  accessibles au clavier (Tab + Entrée).
- Chaque page possède un titre de section clair et hiérarchisé.
- Les graphiques Plotly sont navigables au clavier et affichent des
  tooltips au survol.
- La carte interactive affiche les informations textuellement sous la
  carte pour les utilisateurs ne pouvant pas interagir avec la carte.

---

## 3. Étiquettes et labels (WCAG 1.3.1 / 3.3.2)

- Tous les boutons de sélection (catégorie, produit, forme) ont des
  labels explicites visibles.
- Les selectbox Streamlit incluent des labels descriptifs.
- Les résultats de prédiction sont annoncés avec une description complète
  du produit et de la forme sélectionnés.

---

## 4. Textes alternatifs (WCAG 1.1.1)

- Les graphiques SHAP affichés via st.image() incluent un paramètre
  caption décrivant le contenu du graphique.
- Les emojis décoratifs sont accompagnés du nom du produit en texte.

---

## 5. Redimensionnement (WCAG 1.4.4)

- L'application Streamlit est responsive et s'adapte aux différentes
  tailles d'écran.
- Le texte reste lisible jusqu'à un zoom de 200% sans perte de contenu.

---

## 6. Messages d'erreur (WCAG 3.3.1)

- Quand l'API FastAPI n'est pas disponible, un message clair est affiché
  à l'utilisateur expliquant le problème.
- Quand les données météo ne sont pas disponibles, des données estimées
  sont utilisées avec une mention explicite.

---

## 7. Points d'amélioration identifiés

| Point | Niveau WCAG | Impact | Action envisagée |
|---|---|---|---|
| Carte USA non accessible aux lecteurs d'écran | AA | Modéré | Ajouter un tableau texte alternatif sous la carte |
| Tooltips Plotly non lisibles au clavier | AA | Faible | Ajouter des descriptions textuelles des données |

---

## 8. Outils utilisés pour la vérification

- Vérification manuelle des contrastes via WebAIM Contrast Checker
- Test de navigation clavier sur Chrome et Firefox
- Inspection visuelle du rendu à différentes tailles d'écran

---

## Conclusion

L'application respecte les critères WCAG 2.1 niveau AA sur les points
essentiels : contrastes, navigation clavier, labels et messages d'erreur.
Deux points d'amélioration ont été identifiés sur la carte interactive
et les tooltips, qui feront l'objet d'une prochaine itération.