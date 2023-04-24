import numpy as np
import torch


def get_evaluator(arch):
    if arch == 'densenet':
        return eval_camelyon
    elif arch == 'resnet50':
        return eval_crc


def eval_camelyon(model, ldr, device):
    correct = 0
    batch_total = 0
    model.eval()

    with torch.no_grad():
        for (imgs, labels) in ldr:
            imgs, labels = imgs.to(device), labels.to(device)

            output = model(imgs)

            # # loss = self.criterion(pred, target)
            _, pred = torch.max(output, 1)
            test_correct = pred.eq(labels).sum()

            correct += test_correct.item()
            batch_total += labels.size(0)

    print(f'output nan? {torch.isnan(output).any()}, img_nan?:{torch.isnan(imgs).any()}"')
    acc = 100. * correct / batch_total
    # logger.info(f"Server Test accuracy:{acc}, {correct}/{batch_total} correct")
    return f"Server Test accuracy:{acc}, {correct}/{batch_total} correct"


def eval_crc(model, ldr, device):
    correct = 0
    batch_total = 0
    model.to(device)
    model.eval()

    with torch.no_grad():
        for img, label in ldr:
            img, label = img.type(torch.FloatTensor).to(device).permute(0, 3, 1, 2), label.to(device)

            output = model(img)

            # # loss = self.criterion(pred, target)
            _, pred = torch.max(output, 1)
            test_correct = pred.eq(label).sum()

            correct += test_correct.item()
            batch_total += label.size(0)

    print(f'output nan? {torch.isnan(output).any()}, img_nan?:{torch.isnan(img).any()}"')
    acc = 100. * correct / batch_total
    # logger.info(f"Server Test accuracy:{acc}, {correct}/{batch_total} correct")
    return f"Server Test accuracy:{acc}, {correct}/{batch_total} correct"


def eval_glas(model, ldr, device):
    '''
    swap = (lambda x: np.einsum('bchw->bhwc', x))
    repeat_channel = (lambda x: np.repeat(x, 3, axis=-1))

    #test_output = test(model)

    ori_mask = repeat_channel(swap(np.vstack(tuple(test_output['segs']))))  # binarised masking

    model.to(device)
    model.eval()
    '''
    pass
