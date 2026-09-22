import 'package:flutter/material.dart';

class PlanPoint {
  final double x;
  final double y;
  const PlanPoint(this.x, this.y);
}

class PlanStall {
  final String id;
  final List<PlanPoint> points;
  const PlanStall({required this.id, required this.points});

  factory PlanStall.fromMap(Map<String, dynamic> data) {
    return PlanStall(
      id: data['id']?.toString() ?? '',
      points: [
        for (final point in (data['points'] as List? ?? []))
          PlanPoint(
            (point['x'] as num?)?.toDouble() ?? 0,
            (point['y'] as num?)?.toDouble() ?? 0,
          ),
      ],
    );
  }
}

class ParkingPlan {
  final int rows;
  final int cols;
  final List<List<String>> cells;
  final List<PlanPoint> outline;
  final List<PlanStall> stalls;
  final bool defined;

  const ParkingPlan({
    this.rows = 0,
    this.cols = 0,
    this.cells = const [],
    this.outline = const [],
    this.stalls = const [],
    this.defined = false,
  });

  bool get isMatrix => rows > 0 && cols > 0 && cells.isNotEmpty;

  factory ParkingPlan.fromMap(Map<String, dynamic> data) {
    final rawCells = data['cells'] as List? ?? [];
    final cells = [
      for (final row in rawCells)
        [
          for (final cell in (row as List? ?? [])) cell.toString(),
        ],
    ];
    return ParkingPlan(
      rows: (data['rows'] as num?)?.toInt() ?? 0,
      cols: (data['cols'] as num?)?.toInt() ?? 0,
      cells: cells,
      outline: [
        for (final point in (data['outline'] as List? ?? []))
          PlanPoint(
            (point['x'] as num?)?.toDouble() ?? 0,
            (point['y'] as num?)?.toDouble() ?? 0,
          ),
      ],
      stalls: [
        for (final stall in (data['stalls'] as List? ?? []))
          PlanStall.fromMap(Map<String, dynamic>.from(stall as Map)),
      ],
      defined: data['defined'] == true || cells.isNotEmpty,
    );
  }
}

class ParkingPlanView extends StatelessWidget {
  final ParkingPlan plan;
  final Map<String, bool> occupancy;
  final List<String> occluded;
  final String? selectedId;
  final void Function(String id)? onStallTap;

  const ParkingPlanView({
    super.key,
    required this.plan,
    required this.occupancy,
    this.occluded = const [],
    this.selectedId,
    this.onStallTap,
  });

  @override
  Widget build(BuildContext context) {
    if (plan.isMatrix) {
      return _MatrixPlanView(
        plan: plan,
        occupancy: occupancy,
        occluded: occluded,
        selectedId: selectedId,
        onStallTap: onStallTap,
      );
    }
    if (plan.stalls.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            'Administratzaileak konfigurazioan erabiltzaile-bista (matrizea) definitu behar du.',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }
    return LayoutBuilder(
      builder: (context, constraints) {
        return GestureDetector(
          onTapDown: onStallTap == null
              ? null
              : (details) {
                  final tapped = _hit(
                    details.localPosition,
                    Size(constraints.maxWidth, constraints.maxHeight),
                  );
                  if (tapped != null) onStallTap!(tapped);
                },
          child: CustomPaint(
            size: Size(constraints.maxWidth, constraints.maxHeight),
            painter: _PlanPainter(
              plan: plan,
              occupancy: occupancy,
              occluded: occluded,
              selectedId: selectedId,
            ),
          ),
        );
      },
    );
  }

  String? _hit(Offset pos, Size size) {
    for (final stall in plan.stalls.reversed) {
      final points = [
        for (final p in stall.points) Offset(p.x * size.width, p.y * size.height),
      ];
      if (_inside(pos, points)) return stall.id;
    }
    return null;
  }

  bool _inside(Offset point, List<Offset> polygon) {
    var inside = false;
    for (var i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      final a = polygon[i];
      final b = polygon[j];
      final hit = ((a.dy > point.dy) != (b.dy > point.dy)) &&
          (point.dx < (b.dx - a.dx) * (point.dy - a.dy) / ((b.dy - a.dy) + 0.0001) + a.dx);
      if (hit) inside = !inside;
    }
    return inside;
  }
}

class _MatrixPlanView extends StatelessWidget {
  final ParkingPlan plan;
  final Map<String, bool> occupancy;
  final List<String> occluded;
  final String? selectedId;
  final void Function(String id)? onStallTap;

  const _MatrixPlanView({
    required this.plan,
    required this.occupancy,
    required this.occluded,
    required this.selectedId,
    required this.onStallTap,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(12),
      child: GridView.count(
        crossAxisCount: plan.cols.clamp(1, 20),
        crossAxisSpacing: 8,
        mainAxisSpacing: 8,
        childAspectRatio: 1 / 1.7,
        children: [
          for (final row in plan.cells)
            for (final id in row) _stall(id),
        ],
      ),
    );
  }

  Widget _stall(String id) {
    if (id.isEmpty || id == '~') {
      return DecoratedBox(
        decoration: BoxDecoration(
          color: const Color(0xFF1C1F21),
          borderRadius: BorderRadius.circular(6),
        ),
        child: const Center(
          child: SizedBox(
            width: 28,
            height: 3,
            child: ColoredBox(color: Color(0xFFC9A227)),
          ),
        ),
      );
    }
    final hidden = occluded.contains(id);
    final busy = occupancy[id] == true;
    final free = occupancy[id] == false;
    final color = hidden
        ? const Color(0xFF2A86D4)
        : busy
            ? const Color(0xFFD45B4A)
            : free
                ? const Color(0xFF3CAF6F)
                : const Color(0xFF2A3336);
    return Material(
      color: color.withValues(alpha: 0.35),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(6),
        side: BorderSide(
          color: id == selectedId ? Colors.white : color,
          width: 2,
        ),
      ),
      child: InkWell(
        onTap: onStallTap == null ? null : () => onStallTap!(id),
        child: Column(
          children: [
            const Spacer(),
            if (busy)
              Container(
                width: 28,
                height: 42,
                decoration: BoxDecoration(
                  color: const Color(0xFF23282B),
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
            Padding(
              padding: const EdgeInsets.all(8),
              child: Text(
                id,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PlanPainter extends CustomPainter {
  final ParkingPlan plan;
  final Map<String, bool> occupancy;
  final List<String> occluded;
  final String? selectedId;

  _PlanPainter({
    required this.plan,
    required this.occupancy,
    required this.occluded,
    required this.selectedId,
  });

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = const Color(0xFF0A1816));
    if (plan.outline.length >= 3) {
      _poly(
        canvas,
        size,
        plan.outline,
        fill: const Color(0x142AA7A1),
        stroke: const Color(0xFF2AA7A1),
      );
    }
    for (final stall in plan.stalls) {
      final hidden = occluded.contains(stall.id);
      final busy = occupancy[stall.id] == true;
      final free = occupancy[stall.id] == false;
      final fill = hidden
          ? const Color(0x732A86D4)
          : busy
              ? const Color(0x80D45B4A)
              : free
                  ? const Color(0x803CAF6F)
                  : const Color(0x33B7C4C6);
      final stroke = stall.id == selectedId
          ? Colors.white
          : hidden
              ? const Color(0xFF5EB0FF)
              : busy
                  ? const Color(0xFFD45B4A)
                  : free
                      ? const Color(0xFF3CAF6F)
                      : const Color(0xFF7EE0D8);
      _poly(canvas, size, stall.points, fill: fill, stroke: stroke);
      if (stall.points.isNotEmpty) {
        final origin = Offset(
          stall.points.first.x * size.width + 8,
          stall.points.first.y * size.height + 18,
        );
        final painter = TextPainter(
          text: TextSpan(
            text: stall.id,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        painter.paint(canvas, origin);
      }
    }
  }

  void _poly(
    Canvas canvas,
    Size size,
    List<PlanPoint> points, {
    required Color fill,
    required Color stroke,
  }) {
    if (points.isEmpty) return;
    final path = Path()
      ..moveTo(points.first.x * size.width, points.first.y * size.height);
    for (final point in points.skip(1)) {
      path.lineTo(point.x * size.width, point.y * size.height);
    }
    path.close();
    canvas.drawPath(path, Paint()..color = fill);
    canvas.drawPath(
      path,
      Paint()
        ..color = stroke
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
  }

  @override
  bool shouldRepaint(covariant _PlanPainter oldDelegate) {
    return oldDelegate.plan != plan ||
        oldDelegate.occupancy != occupancy ||
        oldDelegate.occluded != occluded ||
        oldDelegate.selectedId != selectedId;
  }
}
