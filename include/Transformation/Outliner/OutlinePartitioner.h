#pragma once

#include "Analysis/Compatibility/Overlay/CompatibilityOverlayConstVisitor.h"

#include "Analysis/Compatibility/Overlay/CompatibilityOverlay.h"
#include "Analysis/Shape/ShapeAnalysis.h"

namespace Transformation {

class OutlinePartitioner : public Analysis::CompatibilityOverlayConstVisitor
{
public:
	OutlinePartitioner(const Analysis::ShapeAnalysis& shapeAnalysis) : m_shapeAnalysis(shapeAnalysis) {}

	Analysis::CompatibilityOverlay *Partition(const Analysis::CompatibilityOverlay *overlay);

	void Visit(const Analysis::CompatibilityOverlay *overlay) override;

	void Visit(const Analysis::FunctionCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::IfCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::WhileCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::RepeatCompatibilityOverlay *overlay) override;

private:
	unsigned int GetSortingOutDegree(const Analysis::CompatibilityOverlay *overlay, const HorseIR::Statement *statement);
	unsigned int GetSortingOutDegree(const Analysis::CompatibilityOverlay *overlay);

	const Analysis::ShapeAnalysis& m_shapeAnalysis;

	struct PartitionElement
	{
		enum class Kind {
			Statement,
			Overlay
		};

		PartitionElement(Kind kind) : m_kind(kind) {}

		virtual std::size_t Hash() const = 0;

		Kind GetKind() const { return m_kind; }

		bool operator==(const PartitionElement& other) const;
		bool operator!=(const PartitionElement& other) const
		{
			return !(*this == other);
		}

	private:
		Kind m_kind;
	};

	struct StatementElement : PartitionElement
	{
		StatementElement(const HorseIR::Statement *statement) : PartitionElement(PartitionElement::Kind::Statement), m_statement(statement) {}
		StatementElement(const StatementElement& other) : PartitionElement(PartitionElement::Kind::Statement), m_statement(other.m_statement) {}

		const HorseIR::Statement *GetStatement() const { return m_statement; }

		std::size_t Hash() const override
		{
			return std::hash<const HorseIR::Statement *>()(m_statement);
		}

		bool operator==(const StatementElement& other) const
		{
			return (m_statement == other.m_statement);
		}

		bool operator!=(const StatementElement& other) const
		{
			return !(*this == other);
		}

	private:
		const HorseIR::Statement *m_statement = nullptr;
	};

	struct OverlayElement : PartitionElement
	{
		OverlayElement(const Analysis::CompatibilityOverlay *overlay) : PartitionElement(PartitionElement::Kind::Overlay), m_overlay(overlay) {}

		const Analysis::CompatibilityOverlay *GetOverlay() const { return m_overlay; }

		std::size_t Hash() const override
		{
			return std::hash<const Analysis::CompatibilityOverlay *>()(m_overlay);
		}

		bool operator==(const OverlayElement& other) const
		{
			return (m_overlay == other.m_overlay);
		}

		bool operator!=(const OverlayElement& other) const
		{
			return !(*this == other);
		}

	private:
		const Analysis::CompatibilityOverlay *m_overlay = nullptr;
	};

	struct PartitionElementHash
	{
		 bool operator()(const PartitionElement *element) const
		 {
			 return element->Hash();
		 }
	};

	struct PartitionElementEquals
	{
		 bool operator()(const PartitionElement *element1, const PartitionElement *element2) const
		 {
			 return (*element1 == *element2);
		 }
	};

	std::unordered_map<const PartitionElement *, unsigned int, PartitionElementHash, PartitionElementEquals> m_edges;
	Analysis::FunctionCompatibilityOverlay *m_overlay = nullptr;
};

inline bool OutlinePartitioner::PartitionElement::operator==(const PartitionElement& other) const
{
	if (m_kind == other.m_kind)
	{
		switch (m_kind)
		{
			case OutlinePartitioner::PartitionElement::Kind::Statement:
				return (static_cast<const StatementElement&>(*this) == static_cast<const StatementElement&>(other));
			case OutlinePartitioner::PartitionElement::Kind::Overlay:
				return (static_cast<const OverlayElement&>(*this) == static_cast<const OverlayElement&>(other));
		}
	}
	return false;
}

}
