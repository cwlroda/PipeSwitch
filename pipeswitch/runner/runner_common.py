import torch


class ModelSummary:
    def __init__(self, mode, device, model_name, model_class, param_trans_pipe):
        """ """
        self.mode = mode
        self.device = device
        self.model_name = model_name
        self.model_class = model_class
        self.param_trans_pipe = param_trans_pipe

    def execute(self, data_b):
        if data_b is None:
            return self.func(self.device, self.model, self.data_loader)
        else:
            return self.func(self.device, self.model, data_b)

    def reset_initialized(self, mod):
        if hasattr(mod, "initialized"):
            mod.initialized = False
        for child in mod.children():
            self.reset_initialized(child)

    def insert_lock_hook(self, shape_summary_list):
        """ """
        for _, _, _, mod_sublist in shape_summary_list:
            if len(mod_sublist) > 0:
                mod = mod_sublist[0]
                mod.initialized = False

                def hook_wait_for_parameter_lock(mod, *_):
                    if not mod.initialized:
                        complete_name = self.param_trans_pipe.recv()
                        if (
                            hasattr(mod, "fullname")
                            and complete_name != mod.fullname
                        ):
                            raise Exception("Invalid complete trans")
                        mod.initialized = True

                mod.register_forward_pre_hook(hook_wait_for_parameter_lock)

    def insert_terminate_hook(self, mod):
        """ """

        def hook_terminate(*_):
            torch.cuda.synchronize(self.device)

        if len(list(mod.children())) == 0:
            mod.register_forward_hook(hook_terminate)
            mod.register_backward_hook(hook_terminate)
        else:
            for child in mod.children():
                self.insert_terminate_hook(child)

    def load_model(self):
        (
            self.model,
            self.func,
            self.shape_summary_list,
        ) = self.model_class.import_task()

        self.data_loader = self.model_class.import_data_loader()

        # Eliminate parameters and buffers
        self.reset_initialized(self.model)

        # Insert locks for waiting parameters
        # and add pre_formard_hook to wait for locks
        self.insert_lock_hook(self.shape_summary_list)

        # Add hooks for termination in both forward and backward propagations
        if "training" in self.model_name:
            self.insert_terminate_hook(self.model)

        if self.mode == "gpu":
            # Allocate fake memory for parameters
            self.cuda_stream_for_parameter = torch.cuda.Stream(self.device)
            self.cuda_stream_for_computation = torch.cuda.Stream(self.device)

            with torch.cuda.stream(self.cuda_stream_for_parameter):
                torch.cuda.insert_cache_for_param(self.device)
            with torch.cuda.stream(self.cuda_stream_for_computation):
                torch.cuda.insert_cache_for_comp(self.device)

            with torch.cuda.stream(self.cuda_stream_for_parameter):
                for (
                    shape_list,
                    param_list,
                    buf_list,
                    _,
                ) in self.shape_summary_list:
                    for shape, p in zip(
                        shape_list[: len(param_list)], param_list
                    ):
                        p.data = torch.empty(
                            shape, device=f"cuda:{self.device}"
                        )
                    for shape, b in zip(
                        shape_list[len(param_list) :], buf_list
                    ):
                        mod, key = b
                        mod._buffers[key] = torch.empty(
                            shape, device=f"cuda:{self.device}"
                        )
