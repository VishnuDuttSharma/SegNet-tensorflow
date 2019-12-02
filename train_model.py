from SegNet import SegNet
model = SegNet(conf_file="configs/config_bs4_ep40k.json")
model.train(max_steps=40001, batch_size=4)
model.save()


