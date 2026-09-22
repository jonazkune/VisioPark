import 'package:flutter/material.dart';

import '../widgets/parking_grid.dart';

class ParkingSpacesScreen extends StatelessWidget {
  final String parkingId;

  const ParkingSpacesScreen({super.key, required this.parkingId});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Plazak')),
      body: ParkingGrid(parkingId: parkingId),
    );
  }
}
