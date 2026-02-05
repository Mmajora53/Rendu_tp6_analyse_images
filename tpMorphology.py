import cv2
import numpy as np




################### Version pour les images binaires : 

####### Exercice 1 :

def dilation(image, elem_struct) :
    """
    Cette fonction implémente l'opération de dilatation sur une image binaire.
    Cela consite à passer scanner l'image d'entrée avec l'élément structurant
    et si au moins une case à 1 de l'élément structurant passe sur un pixel de
    l'image qui vaut 255, alors le pixel de l'image qui est sous le centre de 
    l'élément structurant aura la valeur 255.
    
    :param image: image d'entrée en niveaux de gris
    :param elem_struct: élément structurant
    """    
    height, width = image.shape #dimensions de l'image
    elem_height, elem_width = elem_struct.shape #dimensions de l'élément structurant

    #serivra au parcours de l'élément structurant
    centre_elemH = elem_height//2
    centre_elemW = elem_width//2

    res = np.zeros_like(image) #sera l'image de sortie

    #pour chaque pixel de l'image d'entrée
    for y in range(height) :
        for x in range(width) :
            
            if image[y,x] == 255 : #si le pixel est à 255, le pixel de l'image de sortie est mis à 255
                res[y,x] = 255

            #pour chaque case de l'élément structurant
            for ey in range(-centre_elemH, centre_elemH+1) :
                for ex in range(-centre_elemW, centre_elemW+1) :

                    #on calcule les index correspondant dans elem_struct
                    iy = ey+centre_elemH
                    ix = ex+centre_elemW

                    #si la case de coordonnée (iy, ix) vaut 1
                    if elem_struct[iy, ix] == 1 :
                        #on calcule les coordonées correspondantes dans l'image
                        newY = y+ey
                        newX = x+ex

                        #si le pixel est dans l'image et qu'il vaut 255
                        if 0<=newY<height and 0<=newX<width and image[newY,newX]==255 :
                            res[y,x] = 255 #pixel de l'image réusltante est mis à 255
    return res





####### Exercice 2 :


def erode(image, elem_struct) :
    """
    Cette fonction implémente l'opération d'érosin sur une image binaire.
    Cela consite à scanner l'image d'entrée avec l'élément structurant
    et si au toutes les cases à 1 de l'élément structurant (et tout l'élément struturant)
    sont contenues dans le voisinage du pixel scanné, alors le pixel de l'image
    qui est sous le centre de l'élément structurant (donc le pixel scanné en question) vaudra 255.
    Sinon, le pixel vaudra 0.
    
    :param image: image d'entrée en niveaux de gris
    :param elem_struct: élément structurant
    """

    height, width = image.shape
    elem_height, elem_width = elem_struct.shape

    centre_elemH = elem_height//2
    centre_elemW = elem_width//2

    #res = np.zeros_like(image)
    res = np.full_like(image, 255)

    for y in range(height) :
        for x in range(width) :

            #if image[y,x] == 0 :
            #    res[y,x] = 0
            
            all_white = True

            for ey in range(-centre_elemH, centre_elemH+1) :
                for ex in range(-centre_elemW, centre_elemW+1) :

                    iy = ey+centre_elemH
                    ix = ex+centre_elemW

                    if elem_struct[iy, ix] == 1 :
                        newY = y+ey
                        newX = x+ex

                        if 0<=newY<height and 0<=newX<width and image[newY,newX]==0 :
                            all_white = False
            
            if not all_white :
                res[y,x] = 0
    
    return res





####### Exercice 3 :

def close(image, elem_struct) :
    """
    Cette fonction implémente la fermeture qui consiste à appliquer
    la dilation sur une image, puis l'érosion sur le résultat de la dilatation.
    
    :param image: image d'entrée en niveaux de gris
    :param elem_struct: élément structurant
    """
    dilated = dilation(image, elem_struct)
    eroded = erode(dilated, elem_struct)

    return eroded



####### Exercice 4 :


def open(image, elem_struct) :
    """
    Cette fonction implémente l'ouverture qui consiste à appliquer
    l'érosion sur une image, puis la dilatation sur le résultat de l'érosion.
    
    :param image: image d'entrée en niveaux de gris
    :param elem_struct: élément structurant
    """
    eroded = erode(image, elem_struct)
    dilated = dilation(eroded, elem_struct)

    return dilated




####### Exercice 5 :


def morphologicalGradient(image, elem_struct) :
    """
    Cette fonction implémente le gradient morphologique qui consiste à appliquer
    la dilation sur une image, puis l'érosion sur l'image d'entrée encore.
    Enfin, on retourne la différence entre la dilatation et l'érosion.
    
    :param image: image d'entrée en niveaux de gris
    :param elem_struct: élément structurant
    """
    dilated = dilation(image, elem_struct)
    eroded = erode(image, elem_struct)

    return dilated - eroded





### Tests pour les images binaires : 



image1 = cv2.imread("/Users/MariaAydin1/Documents/M1_VMI_25_26/semestre2/AnalayseImages/tpHist_ressources/cas5.png", cv2.IMREAD_GRAYSCALE)


#éléments struturants fournis par Grok

# Croix (3x3)
se = np.array([[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]], dtype=np.uint8)

# Disque de rayon ≈ 1 (3×3)
se_disk_3x3 = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

# Disque de rayon ≈ 2 (5×5)
se_disk_5x5 = np.array([
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0]
], dtype=np.uint8)

se_vline_5x1 = np.array([
    [1],
    [1],
    [1],
    [1],
    [1]
], dtype=np.uint8)


test_dilation = dilation(image1, se_disk_3x3)
cv2.imshow("Dilation (binary)", test_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()



test_erode = erode(image1, se_disk_3x3)

cv2.imshow("Erode (binary)", test_erode)
cv2.waitKey(0)
cv2.destroyAllWindows()



test_close = close(image1, se_disk_3x3)

cv2.imshow("Close (binary)", test_close)
cv2.waitKey(0)
cv2.destroyAllWindows()



test_open = open(image1, se_disk_3x3)

cv2.imshow("Open (binary)", test_open)
cv2.waitKey(0)
cv2.destroyAllWindows()


test_morphoGradient = morphologicalGradient(image1, se_disk_3x3)

cv2.imshow("Morphological gradient (binary)", test_morphoGradient)
cv2.waitKey(0)
cv2.destroyAllWindows()


