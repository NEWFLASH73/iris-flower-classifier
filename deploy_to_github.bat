@echo off
echo 🌷 Déploiement du projet Iris sur GitHub...
echo.

:: Vérifier si Git est installé
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git n'est pas installé!
    echo 📥 Téléchargez Git depuis: https://git-scm.com
    pause
    exit /b 1
)

:: Vérifier l'état actuel
echo 🔍 Vérification de l'état Git...
git remote -v
echo.
git status
echo.

:: Configurer Git (remplacez avec vos informations)
echo 📝 Configuration de Git...
git config user.email "newflash73@example.com"
git config user.name "NEWFLASH73"

:: Gérer le remote existant
echo 🔗 Gestion du remote...
git remote remove origin
git remote add origin https://github.com/NEWFLASH73/iris-flower-classifier.git

:: Ajouter les fichiers
echo 📁 Ajout des fichiers...
git add .

:: Faire le commit
echo 💾 Création du commit...
git commit -m "Initial commit: Iris Flower Classification App with Streamlit"

:: Pousser sur GitHub
echo 🚀 Poussée vers GitHub...
git branch -M main
git push -u origin main

if errorlevel 1 (
    echo.
    echo ❌ Erreur lors de la poussée vers GitHub.
    echo 💡 Essayez cette commande manuellement:
    echo   git push -u origin main --force
    pause
    exit /b 1
)

echo.
echo ✅ Projet déployé avec succès sur GitHub!
echo 🌐 Voir votre projet: https://github.com/NEWFLASH73/iris-flower-classifier
echo.
pause