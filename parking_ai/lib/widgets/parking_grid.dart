import 'package:flutter/material.dart';

import '../models/parking.dart';
import '../services/parking_service.dart';
import 'parking_space_widget.dart';

class ParkingGrid extends StatelessWidget {
  final String parkingId;
  final String? selectedId;
  final void Function(String spaceId)? onSpaceTap;

  const ParkingGrid({
    super.key,
    required this.parkingId,
    this.selectedId,
    this.onSpaceTap,
  });

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<Parking>(
      stream: ParkingService.getParkingDetail(parkingId),
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return Center(child: Text('Errorea: ${snapshot.error}'));
        }
        if (!snapshot.hasData) {
          return const Center(child: CircularProgressIndicator());
        }

        final parking = snapshot.data!;
        final spaces = parking.spacesStatus;
        if (spaces.isEmpty) {
          return const Center(child: Text('Ez dago plazarik definituta.'));
        }

        final keys = spaces.keys.toList()..sort();
        return GridView.builder(
          padding: const EdgeInsets.all(12),
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: parking.cols.clamp(1, 8).toInt(),
            crossAxisSpacing: 8,
            mainAxisSpacing: 8,
          ),
          itemCount: keys.length,
          itemBuilder: (context, index) {
            final key = keys[index];
            return ParkingSpaceWidget(
              id: key,
              occupied: spaces[key] ?? false,
              selected: selectedId == key,
              onTap: onSpaceTap == null ? null : () => onSpaceTap!(key),
            );
          },
        );
      },
    );
  }
}
