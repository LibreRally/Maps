using System;

namespace LibreRally.Maps.TileServer;

internal static class BuildingSegmentHeuristics
{
    private const int LowVegetationMatchMinPoints = 200;
    private const double LowVegetationMatchMinHeight = 0.8;
    private const int LowVegetationPromotionMinPoints = 600;
    private const double LowVegetationPromotionMinHeight = 1.0;

    private const int RoofLikeMatchMinPoints = 700;
    private const double RoofLikeMinHeight = 0.5;
    private const double RoofLikeMaxHeight = 3.5;
    private const double RoofLikeMinPlanArea = 80.0;
    private const double RoofLikeMaxPlanArea = 7000.0;
    private const float RoofLikePromotionMinScore = 0.15f;

    public static bool ShouldEvaluateOsmMatch(global::PipelineSegmentPayload segment)
    {
        if (string.Equals(segment.FeatureType, "building", StringComparison.OrdinalIgnoreCase))
            return true;

        if (string.Equals(segment.FeatureType, "low_vegetation", StringComparison.OrdinalIgnoreCase))
            return segment.PointCount >= LowVegetationMatchMinPoints &&
                   GetHeight(segment) >= LowVegetationMatchMinHeight;

        return IsRoofLikeCandidate(segment);
    }

    public static bool CanPromoteToBuilding(global::PipelineSegmentPayload segment, global::SegmentOsmMatchPayload match)
    {
        if (string.Equals(segment.FeatureType, "building", StringComparison.OrdinalIgnoreCase))
            return true;

        if (!match.OsmId.HasValue ||
            !string.Equals(match.MatchStatus, "matched", StringComparison.OrdinalIgnoreCase) ||
            string.IsNullOrWhiteSpace(match.FeatureType))
        {
            return false;
        }

        if (string.Equals(segment.FeatureType, "low_vegetation", StringComparison.OrdinalIgnoreCase))
            return segment.PointCount >= LowVegetationPromotionMinPoints &&
                   GetHeight(segment) >= LowVegetationPromotionMinHeight;

        return IsRoofLikeCandidate(segment) &&
               (match.MatchScore ?? 0f) >= RoofLikePromotionMinScore;
    }

    public static double GetHeight(global::PipelineSegmentPayload segment)
    {
        return segment.BoundsMax.ElementAtOrDefault(2) - segment.BoundsMin.ElementAtOrDefault(2);
    }

    public static double GetPlanArea(global::PipelineSegmentPayload segment)
    {
        var width = segment.BoundsMax.ElementAtOrDefault(0) - segment.BoundsMin.ElementAtOrDefault(0);
        var length = segment.BoundsMax.ElementAtOrDefault(1) - segment.BoundsMin.ElementAtOrDefault(1);
        return Math.Max(width, 0d) * Math.Max(length, 0d);
    }

    private static bool IsRoofLikeCandidate(global::PipelineSegmentPayload segment)
    {
        if (!string.Equals(segment.FeatureType, "ground", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(segment.FeatureType, "high_vegetation", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var height = GetHeight(segment);
        var planArea = GetPlanArea(segment);
        return segment.PointCount >= RoofLikeMatchMinPoints &&
               height >= RoofLikeMinHeight &&
               height <= RoofLikeMaxHeight &&
               planArea >= RoofLikeMinPlanArea &&
               planArea <= RoofLikeMaxPlanArea;
    }
}
