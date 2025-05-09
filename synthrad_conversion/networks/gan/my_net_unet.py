def train_onestep_Unet(unet, unet_opt, loss_function, inputs, labels):
    unet_opt.zero_grad()
    outputs = unet(inputs)
    loss = loss_function(outputs, labels)
    loss.backward()
    unet_opt.step()
    return loss.item(), outputs