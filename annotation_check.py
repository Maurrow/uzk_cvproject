import matplotlib.pyplot as plt
import glob
import pickle

dataset = "aDN_Control"

result_path = f'./annotations/raw/{dataset}/1/df3d_result*.pkl'
pr_path = f'./annotations/raw/{dataset}/1/df3d_result*.pkl'
d = pickle.load(open(glob.glob(pr_path)[0], 'rb'))

image_path = './example_images/{dataset}/camera_{cam_id}_img_{img_id}.jpg'


cam_id, time = 0, 0

plt.imshow(plt.imread(image_path.format(dataset=dataset,cam_id=0,img_id=0)), cmap="grey")
plt.axis('off')
for joint_id in range(19):
    x, y = d['points2d'][cam_id, time][joint_id, 1] * 960, d['points2d'][cam_id, time][joint_id, 0] * 480
    plt.scatter(x, y, c='blue', s=5)
    plt.text(x, y, f'{joint_id}', c='red')

plt.show(block=True)