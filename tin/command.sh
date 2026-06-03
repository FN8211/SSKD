# 以下命令均从 SSKD/ 根目录执行
# 数据默认路径: ./data/tiny-imagenet-200/

# ── 训练 teacher (vgg13) ──
python tin/tin_teacher.py --arch vgg13 --lr 0.05 --gpu-id 0

# ── 训练 student (vgg8)，选其中一种 SS 方法 ──

# 方法1: Contrastive (原论文主方法，旋转视图作为 augmented views)
python tin/tin_student.py --t-path ./experiments_tin/tin_teacher_vgg13 --s-arch vgg8 --lr 0.05 --gpu-id 0

# 方法2: Rotation (旋转角度预测)
python tin/tin_student_rot.py --t-path ./experiments_tin/tin_teacher_vgg13 --s-arch vgg8 --lr 0.05 --gpu-id 0

# 方法3: Jigsaw (拼图还原)
python tin/tin_student_jigsaw.py --t-path ./experiments_tin/tin_teacher_vgg13 --s-arch vgg8 --lr 0.05 --gpu-id 0

# 方法4: Exemplar (实例判别)
python tin/tin_student_examplar.py --t-path ./experiments_tin/tin_teacher_vgg13 --s-arch vgg8 --lr 0.05 --gpu-id 0

# 加速调试用 (缩短 teacher SS head 预训练 + 减少验证频率)
python tin/tin_student_rot.py --t-path ./experiments_tin/tin_teacher_vgg13 --s-arch vgg8 --lr 0.05 --gpu-id 0 --t-epoch 20 --val-interval 5
