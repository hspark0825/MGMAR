"""Settings of the 29 cases of the AAPM CT-MAR final test set."""

# reconstruction field of view (mm)
FOV = {k: 500 for k in range(1, 30)}
FOV.update({4: 650, 5: 650})
FOV.update({k: 360 for k in range(6, 11)})
FOV.update({k: 355 for k in range(18, 21)})

# decoder refinement iterations
REFINE_STEPS = {k: 1000 for k in range(1, 30)}
REFINE_STEPS.update({18: 3000, 19: 3000, 20: 3000})

# value (HU) written into the metal region of the final image
METAL_HU = dict(zip(range(1, 30), [
    1851, 1901, 801, 351, 351, 2751, 2751, 2751, 2751, 2751,
    351, 351, 1651, 1651, 2751, 2751, 2751, 2151, 2151, 2151,
    1651, 1651, 851, 1251, 1651, 1651, 1301, 1301, 1301]))
