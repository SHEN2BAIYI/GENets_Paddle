from model.genet import *
import paddle

model = GENet_light()
print(model)
paddle.summary(model, (1, 3, 192, 192))
Flops = paddle.flops(model, [1, 3, 192, 192], print_detail=True)