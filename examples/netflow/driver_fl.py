import random
import netshare.ray as ray
from netshare.generators.fl_generator import FederatedGenerator 

if __name__ == '__main__':
    # Change to False if you would not like to use Ray
    ray.config.enabled = False
    ray.init(address="auto")

    # Configuration file
    config_file = "config_example_netflow_nodp.json"
    # Specify the number of clients you want to simulate
    num_clients = 5  # 可以根据需要调整客户端数量

    # Initialize the FederatedGenerator with the configuration and client count
    federated_generator = FederatedGenerator(config=config_file, clients=[f"client_{i}" for i in range(num_clients)])

    # Ensure the `work_folder` does not exist to avoid overwrite errors.
    work_folder = '../../fl_results/test-ugr16'
    
    # Train the federated model
    federated_generator.train_federated(rounds=10, local_epochs=5)  # 进行10轮训练，每个客户端5个局部训练周期

    # Generate synthetic data after training
    federated_generator.distribute_global_model()  # 确保每个客户端获得更新后的全局模型
    federated_generator.generate(work_folder=work_folder)  # 生成合成数据

    # Visualize the results
    federated_generator.visualize(work_folder=work_folder)  # 可视化结果

    ray.shutdown()
