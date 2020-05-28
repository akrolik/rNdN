#include "Runtime/DataBuffers/ListBuffer.h"

namespace Runtime {

ListBuffer::ListBuffer(HorseIR::ListType *type, Analysis::ListShape *shape) : DataBuffer(DataBuffer::Kind::List), m_type(type), m_shape(shape)
{

}

ListBuffer::~ListBuffer()
{
	delete m_type;
	delete m_shape;
}

}
