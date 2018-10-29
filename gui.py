import pygame
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

dataset = np.empty((0,2), dtype='f')

def create_data(position):
    (x,y) = position
    r = np.random.uniform(0, 30)
    phi = np.random.uniform(0, 2*np.pi)
    coord = [x + r*np.cos(phi), y + r*np.sin(phi)]
    global dataset
    dataset = np.append(dataset, [coord], axis=0)

radius = 2
color = (0,0,255)
thickness = 0

bg_color = (255,255,255)
(width, heigth) = (640, 480)
screen = pygame.display.set_mode((width, heigth))
pygame.display.set_caption("K-means")
running = True
pushing = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pushing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            pushing = False

    if pushing and np.random.uniform() > .9:
        create_data(pygame.mouse.get_pos())

    screen.fill(bg_color)

    for i, data in enumerate(dataset):
        pygame.draw.circle(screen, color, (int(data[0]), int(data[1])), radius, thickness)

    pygame.display.flip()

pygame.quit()

#colors = np.array(['#377eb8', '#ff7f00', '#4daf4a'])

#test = KMeans(n_clusters=3).fit_predict(dataset)
#plt.figure()
#plt.scatter(dataset[:,0], dataset[:,1], c=colors[test])
#plt.show()
