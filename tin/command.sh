# Run these commands under SSKD/
# data dir: ./data/tiny-imagenet-200/

# ── teacher training (vgg13) ──
python tin/tin_teacher.py --arch vgg13 --lr 0.05 --gpu-id 0

# ── student training (vgg8) ──

# Contrastive
python tin/tin_student_contrastive.py --t-path ./experiments_tin/tin_teacher_vgg13 --s-arch vgg8 --lr 0.05 --gpu-id 0

# Rotation
python tin/tin_student_rot.py --t-path ./experiments_tin/tin_teacher_vgg13 --s-arch vgg8 --lr 0.05 --gpu-id 0

# Jigsaw
python tin/tin_student_jigsaw.py --t-path ./experiments_tin/tin_teacher_vgg13 --s-arch vgg8 --lr 0.05 --gpu-id 0

# Exemplar
python tin/tin_student_examplar.py --t-path ./experiments_tin/tin_teacher_vgg13 --s-arch vgg8 --lr 0.05 --gpu-id 0

# debug
python tin/tin_student_rot.py --t-path ./experiments_tin/tin_teacher_vgg13 --s-arch vgg8 --lr 0.05 --gpu-id 0 --t-epoch 20 --val-interval 5
