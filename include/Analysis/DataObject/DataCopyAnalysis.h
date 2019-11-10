#pragma once

#include <unordered_map>
#include <utility>

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "Analysis/DataObject/DataObjectAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class DataCopyAnalysis : public HorseIR::ConstHierarchicalVisitor
{
public:
	DataCopyAnalysis(const DataObjectAnalysis& objectAnalysis) : m_objectAnalysis(objectAnalysis) {}

	// Analysis input/output

	void Analyze(const HorseIR::Function *function);

	bool ContainsDataCopy(const DataObject *returnObject) const
	{
		return (m_dataCopies.find(returnObject) != m_dataCopies.end());
	}
	const DataObject *GetDataCopy(const DataObject *returnObject) const { return m_dataCopies.at(returnObject); }

	// Copy collection

	bool VisitIn(const HorseIR::AssignStatement *assignS) override;
	bool VisitIn(const HorseIR::CallExpression *call) override;

	// Debug printing

	std::string DebugString(unsigned int indent = 0);

private:
	const DataObjectAnalysis& m_objectAnalysis;

	std::unordered_map<const DataObject *, const DataObject *> m_dataCopies;
};

}
