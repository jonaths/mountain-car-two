import numpy as np


def snapshot(budget, estimator_value, i_episode, items=40):
    """
    Esto deberia guardar una muestra del modelo aprendido con todos las variables de estado
    mas el presupuesto. Habria que modificarlo para que funcionara independientemente del numero
    de variables de estado
    :param budget: el presupuesto actual con el que se corre el experimento
    :param estimator_value: de tensorflow
    :param i_episode: el episodio actual
    :param items: el numero de muestras a generar
    :return:
    """

    # genera las columnas
    x1 = np.linspace(-1.2, 0.6, num=items)
    x2 = np.linspace(-0.07, 0.07, num=items)

    x1x, y1y = np.meshgrid(x1, x2)

    # enumera del 10 a budget incluyedolo
    step = 10
    bs = range(10, budget + step, step)

    # el numero de filas que contiene el arreglo (una para cada combinacion b x1 x2)
    length = items ** 2 * len(bs)

    # el numero de columnas del arreglo (una para cada parametro y la ultima para y)
    columns = 4

    v = np.zeros((length, columns))

    index = 0
    for b in bs:
        v[index * items ** 2: (index + 1) * items ** 2, 0] = np.full((1, items ** 2), b)
        v[index * items ** 2: (index + 1) * items ** 2, 1] = x1x.ravel()
        v[index * items ** 2: (index + 1) * items ** 2, 2] = y1y.ravel()
        index += 1

    # para cada fila de v
    for r in v:
        # estima segun el modelo
        r[-1] = estimator_value.predict(r[0:3])

    # bandera de no existe el archivo
    new = False

    ep_name = "{0:0>4}".format(i_episode)
    b_name = "{0:0>4}".format(budget)
    filename = 'values/b-' + b_name + '_ep-' + ep_name + '.npy'

    try:
        arr = np.load(filename)
        print "arr found", arr.shape
        pass
    except IOError:
        arr = np.zeros((1, length, 4))
        arr[0] = v
        new = True
        print "arr created", arr.shape
        pass

    if new:
        # si es nuevo no hagas nada
        pass
    else:
        # si no es nuevo agregale el ultimo elemento
        arr = np.append(arr, [v], axis=0)

    # guarda el archivo
    np.save(filename, arr)

    return arr
