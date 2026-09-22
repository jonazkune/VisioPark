import 'package:flutter/material.dart';

import '../models/parking.dart';

class ParkingCard extends StatelessWidget {
  final Parking parking;
  final VoidCallback onTap;

  const ParkingCard({super.key, required this.parking, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final status = parking.isCalibrated
        ? (parking.modelTrained ? 'Entrenatuta' : 'Kalibratuta, entrenatu gabe')
        : 'Kalibratu gabe';

    return Card(
      child: ListTile(
        leading: const Icon(Icons.local_parking),
        title: Text(parking.name),
        subtitle: Text(
          '${parking.visibility == 'private' ? 'Pribatua' : 'Publikoa'} · Libre: ${parking.freeSpaces}/${parking.totalSpaces} · $status',
        ),
        trailing: const Icon(Icons.arrow_forward_ios, size: 16),
        onTap: onTap,
      ),
    );
  }
}
