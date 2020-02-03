#pragma once

namespace CUDA {

class Data
{
public:
	virtual void *GetAddress() = 0;
};

}
