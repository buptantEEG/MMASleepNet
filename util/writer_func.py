import torch

def save_config(writer, config):
    dict = vars(config) # return type dict
    for i, (key, value) in enumerate(dict.items()):
        writer.add_text(f'config/{key}', str(value), i)
        # print(i, key, value)

def save_graph(writer, net, dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Save graph
    train_data_sample = next(iter(dataloader))
    X0 = train_data_sample[0][0,:,:,:].type(torch.FloatTensor).to(device)
    X1 = train_data_sample[1][0,:,:].type(torch.FloatTensor).to(device)
    X2 = train_data_sample[2][0,:,:].type(torch.FloatTensor).to(device)
    # train_data_sample[3] = train_data_sample[3].reshape(train_data_sample[3][0],1,train_data_sample[3][1])
    X3 = train_data_sample[3][0,:,:].type(torch.FloatTensor).to(device)
    writer.add_graph(net, input_to_model=[X0,X1,X2,X3])

def save_scalar(writer, scalar_dict, step):
    for key, value in scalar_dict.items():
        writer.add_scalar(key, value, step)
    
    # writer.add_scalar('Loss/train', train_loss, t)
    # writer.add_scalar('Loss/test', test_loss, t)
    # writer.add_scalar('Accuracy/train', train_acc, t)
    # writer.add_scalar('Accuracy/test', test_acc, t)

def save_text(writer, text_dict):
    for key, value in text_dict.items():
        writer.add_text(key, str(value))