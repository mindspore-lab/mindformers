"""
weight Transform
"""
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import load_checkpoint, save_checkpoint
def param_transfor():
    """
    weight Transform
    """
    loadpath = "/home/ma-user/work/提交代码/transformer-master/ckpt_0/gpt-76_10.ckpt" # 加载权重的路径
    savepath = "./saveweight.ckpt" # 保存权重的路径
    param_dict = load_checkpoint(loadpath)
    matmul = P.MatMul()
    temp = param_dict.copy()
    for keya, valuea in param_dict.items():
        if 'fastffnmapping1' in keya:
            a = Tensor(param_dict[keya])
            keyb = keya.replace('fastffnmapping1', 'fastffnmapping2')
            u1 = keya.split('.', -1)
            print('u1', keya)
            print('u2', keyb)
            b = Tensor(param_dict[keyb])
            newkey = keya.replace('fastffnmapping1', 'mapping')
            if u1[len(u1) - 1] == 'weight':
                c = matmul(a, b)
                del temp[keya]
                del temp[keyb]
                newkey = keya.replace('fastffnmapping1', 'mapping')
                temp[newkey] = c
        if 'fastffnmapping2.bias' in keya:
            valuea.name = keya.replace('fastffnmapping2', 'mapping')
            d = valuea
            del temp[keya]
            newkey = keya.replace('fastffnmapping2', 'mapping')
            temp[newkey] = d
        if 'fastffnprojection1' in keya:
            a = Tensor(param_dict[keya])
            keyb = keya.replace('fastffnprojection1', 'fastffnprojection2')
            u1 = keya.split('.', -1)
            #u2 = keyb.split('.', -1)
            print('u1', keya)
            print('u2', keyb)
            b = Tensor(param_dict[keyb])
            newkey = keya.replace('fastffnprojection1', 'projection')
            if u1[len(u1) - 1] == 'weight':
                c = matmul(a, b)
                del temp[keya]
                del temp[keyb]
                newkey = keya.replace('fastffnprojection1', 'projection')
                temp[newkey] = c
        if 'fastffnprojection2.bias' in keya:
            valuea.name = keya.replace('fastffnprojection2', 'projection')
            d = valuea
            del temp[keya]
            newkey = keya.replace('fastffnprojection2', 'projection')
            temp[newkey] = d
    param_dict = temp
    ms_list = []
    for k, v in param_dict.items():
        param_list = {}
        param_list["name"] = k
        param_list["data"] = v
        ms_list.append(param_list)
    save_checkpoint(ms_list, savepath)

if __name__ == "__main__":
    param_transfor()
    