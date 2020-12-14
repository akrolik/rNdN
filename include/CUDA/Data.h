#pragma once

namespace CUDA {

class Data
{
public:
	virtual const void *GetAddress() const = 0;
};

}
